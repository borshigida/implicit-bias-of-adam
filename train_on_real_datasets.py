import argparse
import datetime
import logging
import os
import pickle
import tempfile
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT

from datasets import get_dataset
from optimizers import MyAdam
from resnet_cifar import resnet50, resnet101


# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True
NUMBER_OF_EPOCHS = 500000
PHYSICAL_BATCH_SIZE = 500
SAVE_EVERY = 10


def atomic_pickle(df, filename):
    # Create a temporary file in the same directory as the target file to ensure they are on the same filesystem
    temp_dir = os.path.dirname(filename)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # Pickle the DataFrame to the temporary file
        with os.fdopen(temp_fd, 'wb') as tmp:
            # This ensures the file descriptor is not closed twice
            temp_fd = None
            if isinstance(df, pd.DataFrame):
                pd.to_pickle(df, tmp)
            else:
                pickle.dump(df, tmp)

        # Atomically replace the original file with the temporary file
        os.replace(temp_path, filename)
    finally:
        # Clean up the temporary file if it still exists
        if temp_fd is not None:
            os.close(temp_fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_model(model_name, num_classes=10):
    if model_name == 'vit':
        return SimpleViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=1024,
        )
    if model_name == 'resnet50':
        return resnet50(num_classes=num_classes)
    if model_name == 'resnet101':
        return resnet101(num_classes=num_classes)
    if model_name == 'cnn':
        layers = [
            # First block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten and Dense layers
            nn.Flatten(),
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
            # No Softmax layer needed because CrossEntropyLoss() expects logits
        ]
        return nn.Sequential(*layers)
    raise ValueError(f"Unsupported model {model_name}")


def get_all_gradients_flattened(model):
    return torch.cat([
        grad.flatten()
        for grad in (param.grad for param in model.parameters() if param.grad is not None)
    ])


def get_hyperparameter_string(optimizer_name, lr, beta_1, beta_2, epsilon):
    return f'{optimizer_name}_lr_{lr:.12}_beta1_{beta_1:.12}_beta2_{beta_2:.12}_epsilon_{epsilon:.12}'


def train_and_test_adam(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        model_name: str, dataset_name: str, optimizer_name: str,
        learning_rate: float, beta_1: float, beta_2: float, epsilon: float,
        hours_to_run: int,
        histories_dir='histories', checkpoints_dir='checkpoints',
        resume_from_checkpoint=False,
        loss_type='ce', augment_dataset=True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    train_dataset, test_dataset, num_classes = get_dataset(dataset_name, augment=augment_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=PHYSICAL_BATCH_SIZE, num_workers=8, persistent_workers=True,
        pin_memory=True, prefetch_factor=8,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=PHYSICAL_BATCH_SIZE, num_workers=8, persistent_workers=True,
        pin_memory=True, prefetch_factor=8,
    )

    model = get_model(model_name, num_classes=num_classes).to(device)

    # loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss(reduction='sum')
    elif loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='sum')
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')
    # optimizer
    epsilon_inside_sqrt = True
    if optimizer_name == 'adam':
        use_bias_correction = True
    elif optimizer_name == 'adam_no_bias_correction':
        use_bias_correction = False
    elif optimizer_name == 'adam_eps_outside':
        use_bias_correction = True
        epsilon_inside_sqrt = False
    elif optimizer_name == 'adam_no_bias_correction_eps_outside':
        use_bias_correction = False
        epsilon_inside_sqrt = False
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    optimizer = MyAdam(
        model.parameters(),
        lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon,
        use_bias_correction=use_bias_correction,
        epsilon_inside_sqrt=epsilon_inside_sqrt,
    )

    os.makedirs(histories_dir, exist_ok=True)
    hyperparameter_string = get_hyperparameter_string(optimizer_name, learning_rate, beta_1, beta_2, epsilon)
    history_file_path = os.path.join(histories_dir, hyperparameter_string + '_history.pickle')
    checkpoint_file_path = os.path.join(checkpoints_dir, hyperparameter_string + '.pt')

    initial_epoch_num = 0
    history = []
    if resume_from_checkpoint:
        logging.info('==> Resuming from checkpoint...')
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError('Error: no checkpoint file found!')

        checkpoint = torch.load(checkpoint_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch_num = checkpoint['epoch']

        # Restore history if the file exists
        if os.path.exists(history_file_path):
            history = pd.read_pickle(history_file_path).to_dict('records')

    iterations_start_datetime = datetime.datetime.now()
    logging.warning(
        '%s on %s trained with %s, '
        'loss %s, augment_dataset: %s, hyperparameter_string: %s. Starting iterations from epoch %d',
        model_name,
        dataset_name,
        optimizer_name,
        loss_type,
        augment_dataset,
        hyperparameter_string,
        initial_epoch_num,
    )
    for epoch_num in range(initial_epoch_num, NUMBER_OF_EPOCHS):
        # Initialize accumulators
        train_loss = 0.
        train_accuracy = 0.
        total_samples = 0  # Keep track of the total number of samples processed

        model.train()  # Set the model to training mode
        optimizer.zero_grad(set_to_none=True)  # Reset gradients for the new epoch

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            if loss_type == 'mse':
                target = F.one_hot(labels, num_classes=num_classes).float()  # pylint: disable=not-callable
            else:
                target = labels
            loss = criterion(outputs, target)

            # Backpropagation - but do not apply gradients yet
            loss.backward()

            # Accumulate loss and accuracy
            train_loss += loss.item()
            train_accuracy += (outputs.argmax(dim=1) == labels).float().sum().item()
            total_samples += images.size(0)

        # divide gradients by the number of samples
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= total_samples

        # Calculate the perturbed one-norm of the gradients
        eps = optimizer.param_groups[0]['eps']
        all_gradients_flat = get_all_gradients_flattened(model)
        perturbed_one_norm = torch.sum(torch.sqrt(all_gradients_flat ** 2 + eps))
        two_norm = torch.sqrt(torch.sum(all_gradients_flat ** 2))
        gradient_magnitudes = torch.abs(all_gradients_flat)
        one_norm = torch.sum(gradient_magnitudes)
        mean_gradient_magnitude = torch.mean(gradient_magnitudes)
        std_gradient_magnitude = torch.std(gradient_magnitudes)

        # Apply gradients after accumulating them across all batches
        optimizer.step()

        # Average loss and accuracy over the epoch
        train_loss /= total_samples
        train_accuracy /= total_samples

        # Evaluate on test data
        model.eval()  # Set model to evaluation mode
        test_loss = 0.
        test_accuracy = 0.
        total_samples_test = 0

        with torch.no_grad():  # Disable gradient computation during evaluation
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                if loss_type == 'mse':
                    target = F.one_hot(labels, num_classes=num_classes).float()  # pylint: disable=not-callable
                else:
                    target = labels
                loss = criterion(outputs, target)

                # Accumulate test loss and accuracy
                test_loss += loss.item()
                test_accuracy += (outputs.argmax(dim=1) == labels).float().sum().item()
                total_samples_test += images.size(0)

        # Average test loss and accuracy
        test_loss /= total_samples_test
        test_accuracy /= total_samples_test

        history.append({
            'epoch': epoch_num + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy * 100,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy * 100,
            'perturbed_one_norm': perturbed_one_norm.item(),
            'two_norm': two_norm.item(),
            'one_norm': one_norm.item(),
            'mean_gradient_magnitude': mean_gradient_magnitude.item(),
            'std_gradient_magnitude': std_gradient_magnitude.item(),
        })

        if epoch_num % SAVE_EVERY == 0:
            # Save checkpoint
            if not os.path.isdir(checkpoints_dir):
                os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save({
                'epoch': epoch_num + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file_path)

            # save history
            history_df = pd.DataFrame(history)
            atomic_pickle(history_df, history_file_path)

        end_iter_time = datetime.datetime.now()
        av_time_per_iteration = (end_iter_time - iterations_start_datetime) / (epoch_num - initial_epoch_num + 1)
        left_iter_count = NUMBER_OF_EPOCHS - epoch_num + 1
        estimated_completion_time = end_iter_time + av_time_per_iteration * left_iter_count

        logging.info(
            'Finishing epoch %d out of %d, '
            'train loss: %.4f, train acc: %.2f%%, test loss: %.4f, '
            'test acc: %.2f%%, pert 1-norm: %.8f, '
            'av time per iter: %s, est completion time: %s',
            epoch_num + 1, NUMBER_OF_EPOCHS, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100,
            perturbed_one_norm, str(av_time_per_iteration), str(estimated_completion_time),
        )

        if (
                end_iter_time - iterations_start_datetime
                > datetime.timedelta(hours=hours_to_run) - 5 * av_time_per_iteration
        ):
            logging.info('Close to time limit, finishing iterations')
            break


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument(
        '--optimizer_name', type=str,
        default='adam',
        choices=['adam', 'adam_no_bias_correction', 'adam_eps_outside', 'adam_no_bias_correction_eps_outside'],
    )
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--beta_1', type=float, required=True)
    parser.add_argument('--beta_2', type=float, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument(
        '--hours_to_run', type=int, default=24,
        help='Number of hours to run the training for (the training will stop shortly before the time limit)',
    )
    parser.add_argument(
        '--histories_dir', type=str, default='histories',
        help='Directory to save the training history to. '
             'The training history is a pickled pandas dataframe with columns: epoch, train_loss, train_accuracy, '
             'test_loss, test_accuracy, perturbed_one_norm, mean_gradient_magnitude, etc.',
    )
    parser.add_argument(
        '--checkpoints_dir', type=str, default='checkpoints',
        help='Directory to save the model checkpoints to. '
             f'Checkpoints are saved and histories are pickled every {SAVE_EVERY} epochs'
    )
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--loss_type', type=str, default='ce')
    parser.add_argument(
        '--augment_dataset', action='store_true',
        help='Whether to use random data augmentation (crop, flip)'
    )

    args = parser.parse_args()

    train_and_test_adam(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
        hours_to_run=args.hours_to_run,
        histories_dir=args.histories_dir,
        checkpoints_dir=args.checkpoints_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        loss_type=args.loss_type,
        augment_dataset=args.augment_dataset,
    )


if __name__ == '__main__':
    main()
