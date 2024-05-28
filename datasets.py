import numpy as np

import torch
from torchvision import datasets, transforms


def calc_mean_and_std(dataset_name):
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'eurosat':
        dataset = datasets.EuroSAT(root='./data', download=True, transform=transforms.ToTensor())
        train_size = int(0.8 * len(dataset))  # 80% for training
        test_size = len(dataset) - train_size  # 20% for testing
        train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    # Use the training part of the dataset for mean and std calculation
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    imgs, _ = next(iter(loader))
    imgs = imgs.numpy()

    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()

    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()

    print(f"Mean: {mean_r:.8f}, {mean_g:.8f}, {mean_b:.8f}")
    print(f"Std: {std_r:.8f}, {std_g:.8f}, {std_b:.8f}")
    return np.array([mean_r, mean_g, mean_b]), np.array([std_r, std_g, std_b])


def get_dataset(dataset_name, augment=True):
    if dataset_name == 'cifar10':
        data_means, data_stds = calc_mean_and_std('cifar10')
    elif dataset_name == 'cifar100':
        data_means, data_stds = calc_mean_and_std('cifar100')
    elif dataset_name == 'eurosat':
        data_means, data_stds = calc_mean_and_std('eurosat')
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    transformations = [transforms.Resize((32, 32))]
    if augment:
        transformations += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
    transformations += [
        transforms.ToTensor(),
        transforms.Normalize(mean=data_means, std=data_stds),
    ]
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_means, std=data_stds),
        ]
    )

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset_name == 'eurosat':
        full_dataset = datasets.EuroSAT(root='./data', download=True, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    # truncation
    # train_subset_indices = np.random.choice(len(train_dataset), 2000, replace=False)
    # test_subset_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    # train_dataset = Subset(train_dataset, train_subset_indices)
    # test_dataset = Subset(test_dataset, test_subset_indices)

    return train_dataset, test_dataset, num_classes
