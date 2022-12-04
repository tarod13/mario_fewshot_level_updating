import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader


def load_numpy_VAE_dataset(
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
    test_tensor = th.from_numpy(val_data).type(th.float)

    return train_tensor, test_tensor

def load_pytorch_VAE_dataset(
    shuffle_seed=0,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load("./data/vae_dataset_test.pt").type(th.float)
    test_tensor = th.load("./data/vae_dataset_test.pt").type(th.float)

    # Shuffling the training data
    th.manual_seed(shuffle_seed)
    idx = th.randperm(train_tensor.size(0))
    train_tensor = train_tensor[idx]

    return train_tensor, test_tensor

def load_pytorch_TN_dataset(
    shuffle_seed=0,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load("./data/tnet_dataset_test.pt").type(th.float)
    test_tensor = th.load("./data/tnet_dataset_test.pt").type(th.float)

    # Shuffling the training data
    th.manual_seed(shuffle_seed)
    idx = th.randperm(train_tensor.size(1))
    train_tensor = train_tensor[:,idx]
    
    return train_tensor, test_tensor

def estimate_token_frequency(
    train_tensor: np.ndarray,
):
    '''Calculates sample frequency for each token.'''
    # Load data
    labels = train_tensor.argmax(dim=1).flatten()
    unique, counts = labels.unique(return_counts=True)
    frequencies = counts / counts.sum()
    return frequencies

def generate_VAE_dataloader(
    batch_size: int = 64, 
    train_percentage: float = 0.8,
    shuffle_train: bool = False,
    shuffle_val: bool = False,
    seed: int = 0,
    load_original: bool = True,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    if load_original:
        train_tensor, val_tensor = load_pytorch_VAE_dataset(seed)
    else:
        train_tensor, val_tensor = load_numpy_VAE_dataset(train_percentage, seed)

    # Create dataloaders
    train_dataset = TensorDataset(train_tensor)
    mario_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataset = TensorDataset(val_tensor)
    mario_val = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle_val)

    # Calculate token frequency
    token_frequencies = estimate_token_frequency(train_tensor)

    return mario_train, mario_val, token_frequencies, train_tensor.shape[1:]


if __name__ == "__main__":

    load_torch = True
    seed = 0

    if not load_torch:
        train_percentage = 0.8
        train_tensor_vae, val_tensor_vae = load_numpy_VAE_dataset(train_percentage, seed)
    else:
        train_tensor_vae, val_tensor_vae = load_pytorch_VAE_dataset(seed)
        train_tensor_tn, val_tensor_tn = load_pytorch_TN_dataset(seed)
        print(f'Shape of TNet train tensor: {train_tensor_tn.shape}')


    total_tensor = th.cat([train_tensor_vae, val_tensor_vae], axis=0)
    frequencies = estimate_token_frequency(total_tensor)
    print('Token appearance percentage:')
    print((frequencies.numpy()*100).round(2))
