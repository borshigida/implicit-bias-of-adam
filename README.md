# On the Implicit Bias of Adam

This repository contains source code for the ICML 2024 paper “On the Implicit Bias of Adam” by M. D. Cattaneo, J. M. Klusowski, B. Shigida. 

## Quick start

The script `train_on_real_datasets.py` trains a neural network for a specified (integer) number of hours and produces two files: the last model checkpoint and the history file where the training metrics are located (a pickled pandas dataframe with columns like epoch, train_loss, train_accuracy, test_loss, test_accuracy, perturbed_one_norm, mean_gradient_magnitude). For example, one could run
```
python train_on_real_datasets.py --model_name resnet50 --dataset_name cifar10 --learning_rate 1e-4 --beta_1 0.9 --beta_2 0.95 --epsilon 1e-7 --hours_to_run 24
```
and wait for it to finish (24 hours). Then, the loss curves and other metrics can be plotted using the file that is located in the `./histories` directory. The dataset is downloaded automatically by torchvision into the `./data` directory. The script calculates the average time it takes to complete one iteration and uses this number to stop training a few iterations before the time limit provided by the `--hours_to_run` argument. 

For a more detailed usage description, please run `python train_on_real_datasets.py -h`. 
