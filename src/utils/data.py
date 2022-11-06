import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader


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
    train_data = data[:train_index, :, :, :]
    val_data = data[train_index:, :, :, :]
    train_tensors = th.from_numpy(train_data).type(th.float)
    test_tensors = th.from_numpy(val_data).type(th.float)

    return train_tensors, test_tensors

def generate_dataset(
    batch_size: int = 64, 
    train_percentage: float = 0.8,
    shuffle_train: bool = False,
    shuffle_val: bool = False,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    train_tensors, val_tensors = load_data(train_percentage)

    # Create dataloaders
    train_dataset = TensorDataset(train_tensors)
    mario_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataset = TensorDataset(val_tensors)
    mario_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val)

    return mario_train, mario_val
