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
    train_percentage=0.8,
    token_hidden='q_mark',
    finetuning=False,
    train=True,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load(f"./data/vae_dataset_{'train' if train else 'test'}_no_{token_hidden}{'' if not finetuning else '_finetuning'}.pt").type(th.float)
    # test_tensor = th.load("./data/vae_dataset_test.pt").type(th.float)

    if not finetuning:
        n_frames = 551 
    else:
        if token_hidden == 'q_mark':
            n_frames = 152
        elif token_hidden == 'cannon':
            n_frames = 148
        elif token_hidden == 'coin':
            n_frames = 228
        else:
            raise ValueError('Invalid token')
    train_index = int(n_frames * train_percentage)

    train_tensor = train_tensor.reshape(n_frames,-1,*train_tensor.shape[1:])
    val_tensor = train_tensor[train_index:].reshape(-1,*train_tensor.shape[2:])
    train_tensor = train_tensor[:train_index].reshape(-1,*train_tensor.shape[2:])
    
    # Shuffling the training data
    th.manual_seed(shuffle_seed)
    idx = th.randperm(train_tensor.size(0))
    train_tensor = train_tensor[idx]

    return train_tensor, val_tensor

def collapse_1_2(x):
    x = x.reshape(x.shape[0],-1,*x.shape[3:])
    return x

def final_reshape(x):
    x = collapse_1_2(x)
    x = x.transpose(0,1)
    return x

def append_transformation_examples(x, n_examples, test: bool = False):
    frame_difference = (th.absolute(x[0] - x[1]).sum((2,3,4)) > 0).float()
    mask = th.where(
        frame_difference != 0, 
        th.ones_like(frame_difference), 
        th.zeros_like(frame_difference),
    )
    mask /= mask.sum(dim=1, keepdim=True)

    sample = Categorical(probs=mask)\
        .sample((mask.shape[1], n_examples))\
        .transpose(1,2).transpose(0,1)

    indices_with_difference = frame_difference.nonzero(as_tuple=True)
    indices_with_no_difference = (1-frame_difference).nonzero(as_tuple=True)

    # With difference
    origin_with_difference = x[0][indices_with_difference]
    target_with_difference = x[1][indices_with_difference]
    x_with_difference = th.stack([
        origin_with_difference, target_with_difference
    ], dim=0)

    sample_with_difference = sample[indices_with_difference]
    indices_where_example_equal_to_pair = (
        sample_with_difference==indices_with_difference[1].reshape(-1,1)
    )
    sample_with_difference = th.where(
        indices_where_example_equal_to_pair, 
        -th.ones_like(sample_with_difference), 
        sample_with_difference
    )
    sample_with_difference = th.sort(sample_with_difference, dim=1)[0]
    n_examples_to_remove = indices_where_example_equal_to_pair.sum(1).max()    

    # With no difference
    n_examples_with_difference = x_with_difference.shape[1]
    origin_with_no_difference = x[0][indices_with_no_difference][:n_examples_with_difference]
    target_with_no_difference = x[1][indices_with_no_difference][:n_examples_with_difference]
    x_with_no_difference = th.stack([
        origin_with_no_difference, target_with_no_difference
    ], dim=0)

    sample_with_no_difference = sample[indices_with_no_difference]
    indices_where_example_equal_to_pair = (
        sample_with_no_difference==indices_with_no_difference[1].reshape(-1,1)
    )
    sample_with_no_difference = th.where(
        indices_where_example_equal_to_pair, 
        -th.ones_like(sample_with_no_difference), 
        sample_with_no_difference
    )
    sample_with_no_difference = th.sort(sample_with_no_difference, dim=1)[0]

    n_examples_to_remove = max(
        n_examples_to_remove, 
        indices_where_example_equal_to_pair.sum(1).max()
    )
    n_examples = n_examples - n_examples_to_remove

    sample_with_difference = sample_with_difference[:,n_examples_to_remove:]
    sample_with_no_difference = \
        sample_with_no_difference[:n_examples_with_difference,n_examples_to_remove:]

    settings_with_difference = th.repeat_interleave(
        indices_with_difference[0], n_examples)
    settings_with_no_difference = th.repeat_interleave(
        indices_with_no_difference[0][:n_examples_with_difference], 
        n_examples)
    
    indices_example_with_difference = (
        settings_with_difference.reshape(-1), 
        sample_with_difference.reshape(-1)
    )
    indices_example_with_no_difference = (
        settings_with_no_difference.reshape(-1), 
        sample_with_no_difference.reshape(-1)
    )
    
    x_origin_example_with_difference = x[0][indices_example_with_difference]
    x_target_example_with_difference = x[1][indices_example_with_difference]
    x_example_with_difference = th.stack([
        x_origin_example_with_difference, 
        x_target_example_with_difference
    ], dim=0)

    x_origin_example_with_no_difference = x[0][indices_example_with_no_difference]
    x_target_example_with_no_difference = x[1][indices_example_with_no_difference]
    x_example_with_no_difference = th.stack([
        x_origin_example_with_no_difference, 
        x_target_example_with_no_difference
    ], dim=0)

    x_with_difference = th.repeat_interleave(
        x_with_difference, n_examples, dim=1
    )
    x_with_no_difference = th.repeat_interleave(
        x_with_no_difference, n_examples, dim=1
    )

    x_with_difference = th.cat([
        x_with_difference, x_example_with_difference], dim=0)
    x_with_no_difference = th.cat([
        x_with_no_difference, x_example_with_no_difference], dim=0)

    repeat_times = 1 + x_with_difference.shape[1] // x_with_no_difference.shape[1]
    x_with_no_difference = th.repeat_interleave(
        x_with_no_difference, repeat_times, dim=1
    )[:,:x_with_difference.shape[1]]

    x = th.stack([x_with_difference, x_with_no_difference], dim=1)
    x = x.reshape(*x.shape[0:2],-1,n_examples,*x.shape[3:])
    x = x.transpose(1,2)

    return x

def load_pytorch_TN_dataset(
    train_percentage=0.8,
    shuffle_seed=0,
    n_examples=10,
    token_hidden='q_mark',
    leave_only_one_example=False,
    finetuning=False,
    train=True,
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load(f"./data/tnet_dataset_{'train' if train else 'test'}_no_{token_hidden}{'' if not finetuning else '_finetuning'}.pt").type(th.float)
    
    train_tensor = append_transformation_examples(train_tensor, n_examples)

    n_frames = train_tensor.shape[1]
    train_index = int(n_frames * train_percentage)

    val_tensor = train_tensor[:,train_index:]
    train_tensor = train_tensor[:,:train_index]

    val_tensor = collapse_1_2(val_tensor)
    train_tensor = collapse_1_2(train_tensor)
    
    if leave_only_one_example:
        val_tensor = val_tensor[:,:,0:1]
        train_tensor = train_tensor[:,:,0:1]
    
    val_tensor = final_reshape(val_tensor)
    train_tensor = final_reshape(train_tensor)

    # print(th.absolute(val_tensor[:,0]-val_tensor[:,1]).sum())
    # raise RuntimeError('ehem')
    
    # Shuffling the training data
    th.manual_seed(shuffle_seed)
    idx = th.randperm(train_tensor.shape[0])
    train_tensor = train_tensor[idx] 
    
    return train_tensor, val_tensor

def load_pytorch_TN_dataset_for_finetuning(
    train_percentage=0.8,
    shuffle_seed=0,
    n_examples=10,
    test_percentage=0.1,
    token_hidden='q_mark',
):
    """Returns two tensors with training and validation data."""
    # Loading the data.
    train_tensor = th.load(f"./data/tnet_dataset_train_no_{token_hidden}.pt").type(th.float)
    # test_tensor = th.load(f"./data/tnet_dataset_test_no_{token_hidden}.pt").type(th.float)

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
    token_hidden: str = 'q_mark',
    finetuning: bool = False,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    if load_original:
        train_tensor, val_tensor = load_pytorch_VAE_dataset(
            seed, train_percentage, token_hidden, finetuning)
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
    token_hidden: str = 'q_mark',
    finetuning: bool = False,
):
    '''Generates training and validation DataLoaders.'''
    # Load data
    train_tensor, val_tensor = load_pytorch_TN_dataset(
        train_percentage, seed, n_examples, token_hidden, 
        finetuning=finetuning
    )[:2]

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
    token_hidden = 'q_mark'
    train_percentage = 0.8

    if not load_torch:        
        train_tensor_vae, val_tensor_vae = load_numpy_VAE_dataset(train_percentage, seed)
    else:
        train_tensor_vae, val_tensor_vae = load_pytorch_VAE_dataset(seed, train_percentage, token_hidden)
        train_tensor_tn, val_tensor_tn, test_tensor_tn = load_pytorch_TN_dataset(
            train_percentage, seed, token_hidden=token_hidden)
        print(f'Shape of TNet train tensor: {train_tensor_tn.shape}')

    total_tensor = th.cat([train_tensor_vae, val_tensor_vae], axis=0)
    frequencies = estimate_token_frequency(train_tensor_vae)
    print('Token appearance percentage:')
    print((frequencies.numpy()*100))
