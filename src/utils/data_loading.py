import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
import wandb


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
    train_tensor = th.load("./data/vae_dataset_train.pt").type(th.float)
    test_tensor = th.load("./data/vae_dataset_test.pt").type(th.float)

    # Shuffling the training data
    th.manual_seed(shuffle_seed)
    idx = th.randperm(train_tensor.size(0))
    train_tensor = train_tensor[idx]

    return train_tensor, test_tensor

def final_reshape(x):
    x = x.reshape(x.shape[0],-1,*x.shape[3:])
    x = x.transpose(0,1)
    return x

def append_transformation_examples(x, n_examples, test: bool = False):
    frame_difference = th.absolute(x[0] - x[1]).sum((2,3,4))
    mask = th.where(
        frame_difference != 0, 
        th.ones_like(frame_difference), 
        th.zeros_like(frame_difference),
    )
    mask /= mask.sum(dim=1, keepdim=True)

    sample = Categorical(probs=mask)\
        .sample((mask.shape[1], n_examples))\
        .transpose(1,2).transpose(0,1)
    n_settings = sample.shape[0]
    sample_flatten = sample.view(n_settings, -1)
    example_list = []
    for setting in range(n_settings):
        example_list.append(
            x[:,setting,sample_flatten[setting,:]]
        )
    examples = th.stack(example_list, dim=1)
    x = th.repeat_interleave(x, n_examples, dim=2)
    x = th.cat([x, examples], dim=0)
    if not test:
        x = final_reshape(x)
    return x

def load_pytorch_TN_dataset(
    train_percentage=0.8,
    shuffle_seed=0,
    n_examples=10,
    test_percentage=0.1,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load("./data/tnet_dataset_train.pt").type(th.float)
    test_tensor = th.load("./data/tnet_dataset_test.pt").type(th.float)

    n_frames = train_tensor.shape[2]
    th.manual_seed(shuffle_seed)
    idx = th.randperm(n_frames)
    train_tensor = train_tensor[:,:,idx]
    test_tensor = test_tensor[:,:,idx]
    train_index = int(n_frames * train_percentage)
    val_tensor = train_tensor[:,:,train_index:,:,:,:]
    train_tensor = train_tensor[:,:,:train_index,:,:,:]

    train_tensor = append_transformation_examples(train_tensor, n_examples)
    val_tensor = append_transformation_examples(val_tensor, n_examples)
    test_tensor = append_transformation_examples(test_tensor, n_examples, test=True)

    test_index = int(n_frames * test_percentage)   # should be less than train_index
    test_train_tensor = test_tensor[:,:,:test_index]
    test_val_tensor = test_tensor[:,:,test_index:]    

    test_train_tensor = final_reshape(test_train_tensor)
    test_val_tensor = final_reshape(test_val_tensor)

    # append % of test to train and val
    train_tensor = th.cat([train_tensor, test_train_tensor], dim=0)
    val_tensor = th.cat([val_tensor, test_val_tensor], dim=0)

    # shuffle again
    n_frames = train_tensor.shape[2]
    idx = th.randperm(n_frames)
    train_tensor = train_tensor[:,:,idx]

    return train_tensor, val_tensor, test_val_tensor

def estimate_token_frequency(
    train_tensor: np.ndarray,
):
    '''Calculates sample frequency for each token.'''
    # Load data
    counts = train_tensor.sum((0,2,3)).flatten() + 1
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
        train_tensor, val_tensor = load_numpy_VAE_dataset(
            train_percentage, seed)

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

def generate_TN_dataloader(
    batch_size: int = 64, 
    train_percentage: float = 0.8,
    shuffle_train: bool = False,
    shuffle_val: bool = False,
    seed: int = 0,
    n_examples: int = 10,
    load_original: bool = True,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    train_tensor, val_tensor = load_pytorch_TN_dataset(
        train_percentage, seed, n_examples)[:2]

    # Create dataloaders
    train_dataset = TensorDataset(train_tensor)
    mario_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataset = TensorDataset(val_tensor)
    mario_val = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle_val)

    return mario_train, mario_val, train_tensor.shape[2:]


if __name__ == "__main__":

    load_torch = True
    seed = 0

    if not load_torch:
        train_percentage = 0.8
        train_tensor_vae, val_tensor_vae = load_numpy_VAE_dataset(train_percentage, seed)
    else:
        train_tensor_vae, val_tensor_vae = load_pytorch_VAE_dataset(seed)
        train_tensor_tn, val_tensor_tn, test_tensor_tn = load_pytorch_TN_dataset(
            shuffle_seed=seed)
        print(f'Shape of TNet train tensor: {train_tensor_tn.shape}')

    total_tensor = th.cat([train_tensor_vae, val_tensor_vae], axis=0)
    frequencies = estimate_token_frequency(train_tensor_vae)
    print('Token appearance percentage:')
    print((frequencies.numpy()*100))
