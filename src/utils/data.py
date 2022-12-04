import os
import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
from itertools import combinations
from copy import deepcopy
from collections import defaultdict
import logging


def load_data(
    train_percentage=0.8,
    shuffle_seed=0,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("./data/all_levels_onehot.npz")["levels"]
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into train and test.
    n_data, _, _, _ = data.shape
    train_index = int(n_data * train_percentage)
    train_data = data[:train_index, :-1, :, :]
    val_data = data[train_index:, :-1, :, :]
    train_tensor = th.from_numpy(train_data).type(th.float)
    test_tensors = th.from_numpy(val_data).type(th.float)

    return train_tensor, test_tensors

def estimate_token_frequency(
    train_tensor: np.ndarray,
):
    '''Calculates sample frequency for each token.'''
    # Load data
    labels = train_tensor.argmax(dim=1).flatten()
    unique, counts = labels.unique(return_counts=True)
    frequencies = counts / counts.sum()
    return frequencies

def generate_dataset(
    batch_size: int = 64, 
    train_percentage: float = 0.8,
    shuffle_train: bool = False,
    shuffle_val: bool = False,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    train_tensor, val_tensor = load_data(train_percentage)

    # Create dataloaders
    train_dataset = TensorDataset(train_tensor)
    mario_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataset = TensorDataset(val_tensor)
    mario_val = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle_val)

    # Calculate token frequency
    token_frequencies = estimate_token_frequency(train_tensor)

    return mario_train, mario_val, token_frequencies


if __name__ == "__main__":
    train_percentage = 0.8
    train_tensor, val_tensor = load_data(train_percentage)
    total_tensor = th.cat([train_tensor, val_tensor], axis=0)
    frequencies = estimate_token_frequency(total_tensor)
    print(frequencies)
