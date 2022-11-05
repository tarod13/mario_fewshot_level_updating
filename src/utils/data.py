import numpy as np
import torch as th

def load_data(
    training_percentage=0.8,
    shuffle_seed=0,
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("./data/all_levels_onehot.npz")["levels"]
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = th.from_numpy(training_data).type(th.float)
    test_tensors = th.from_numpy(testing_data).type(th.float)

    return training_tensors, test_tensors
