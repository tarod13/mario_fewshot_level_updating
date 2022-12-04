import os
import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
from itertools import combinations


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


def level_to_list(level_txt):
    '''Returns a list by splitting the level text by \n.'''
    as_list = level_txt.split("\n")
    return [list(row) for row in as_list if row != ""]

def level_to_array(level_list):
    '''Returns an np array'''
    result = np.array(level_list, dtype=(int))
    return (result)

def level_tokenizer(original_rows, number_rows, dictionary):
    for level_row in original_rows:
        for i in range(len(level_row)):
            token = level_row[i]
            number = dictionary[token]
            level_row[i] = number
        number_rows.append(level_to_array(level_row))
        
def narray_to_tensor(
    level_narray, tensor_list, 
    size: int = 20, 
    stride: int = 5, 
    num_classes: int = 13
):
    level_tensor = th.from_numpy(level_narray)
    level_tensor = one_hot(level_tensor.to(th.int64), num_classes=num_classes)
    level_tensor = level_tensor.unfold(1, size, stride)
    tensor_list.append(level_tensor)
    
def unify_save_tensor(tensor_list, file):
    unified_tensor_list = th.cat(tensor_list, dim=1)\
        .transpose(0,1).transpose(1,2)
    print(unified_tensor_list.shape)
    np.savez(file, unified_tensor_list)

def find_possible_transformations(transformation_basis: list, r: int):
    possible_transformations = list(combinations(transformation_basis, r=r))
    possible_transformations = list(set([
        tuple(set(transform)) for transform in possible_transformations
    ]))
    possible_transformations.sort(key=len)
    return possible_transformations

def update_keys_of_dict(old_dict, key_dict):
    new_dict = old_dict.copy()
    for k, v in key_dict.items():
        if k in new_dict:
            new_dict[k] = v
    return new_dict

def generate_token_dicts(token_dicts, transformation_dict):
    transformation_pairs = []
    for transform_key, transform_value in transformation_dict.items():
        token_dicts_to_append = dict()
        for d_key, d_value in token_dicts.items():
            if 'original' in d_key:
                new_d_key = frozenset((transform_key,))
            else:
                new_d_key = frozenset.union(d_key, frozenset((transform_key,)))
            transformation_pairs.append((new_d_key, d_key))
            new_d_value = update_keys_of_dict(d_value, transform_value)
            token_dicts_to_append[new_d_key] = new_d_value
        token_dicts = token_dicts | token_dicts_to_append   # Python 3.9 and onwards
    return token_dicts, transformation_pairs

def complete_transformation_pairs(transformation_pairs):
    transformation_path_dict = {}
    for origin, end in transformation_pairs:
        transformation_path_dict[origin] = [end]
        if len(origin) > 1:
            for element in origin:
                potential_end = frozenset(set(origin).difference(set((element,))))
                if potential_end != end:
                    transformation_path_dict[origin].append(potential_end)
    return transformation_path_dict

def simplify_token_dict_keys(token_dicts, transformation_path_dict):
    remaining_keys = list(token_dicts.keys())
    renamed_token_dicts = {}
    renamed_pairs = []
    for origin, ends in transformation_path_dict.items():
        origin_str = str.join('-', list(origin))
        if origin in remaining_keys:
            remaining_keys.remove(origin)            
            renamed_token_dicts[origin_str] = token_dicts[origin]
        for end in ends:
            end_str = str.join('-', list(end))
            renamed_pairs.append((origin_str, end_str))
            if end in remaining_keys:
                remaining_keys.remove(end)                
                renamed_token_dicts[end_str] = token_dicts[end]
    return renamed_token_dicts, renamed_pairs


def generate_dataset_from_original_levels(
    path: str = './data/basic_13_tokens'
):  
    tensor_list = []
    replaced_tensor_list = []
    
    # Basic representation
    token_dict = {
        "X": "0","S": "1","-": "2",
        "?": "3","Q": "4","E": "5",
        "<": "6",">": "7","[": "8",
        "]": "9","o": "10","b": "11",
        "B": "12"
    }
    
    token_dicts = {
        frozenset(('original',)): token_dict,
    }

    transformation_dict = {
        'no_q_mark': {'Q':'S','?':'S'},
        'no_cannon': {'b':'E','B':'E'},
        'no_coins': {'o':'-'},
    }

    token_dicts, transformation_pairs = generate_token_dicts(
        token_dicts, transformation_dict
    )    
    transformation_path_dict = complete_transformation_pairs(
        transformation_pairs
    )

    token_dicts, transformation_pairs = simplify_token_dict_keys(
        token_dicts, transformation_path_dict
    )

    print(len(transformation_pairs))
    
    # tokens_to_replace = {'?', 'Q'}
    # replace_with = 'S'

    # for i in tokens_to_replace:
    #     token_dict_replaced[i] = token_dict[replace_with]

    # for levelFile in os.listdir(path):
    #     print ("Processing: " + levelFile)   #print level being loaded
    #     with open(path+ '/' + levelFile) as fp:
    #         level_txt = fp.read()
    #         list_rows = level_to_list(level_txt)
    #         list_rows_replaced = level_to_list(level_txt)
            
    #     list_rows_tf_replaced = []
    #     level_tokenizer(list_rows_replaced, list_rows_tf_replaced, token_dict_replaced) 
    #     level_narray_replaced = level_to_array(list_rows_tf_replaced)
    #     narray_to_tensor(level_narray_replaced, replaced_tensor_list)
        
    #     list_rows_tf = []
    #     level_tokenizer(list_rows, list_rows_tf, token_dict)
    #     level_narray = level_to_array(list_rows_tf)
    #     narray_to_tensor(level_narray, tensor_list)

    # unify_save_tensor(tensor_list, './data/original_data.npz')
    # unify_save_tensor(replaced_tensor_list, './data/replaced_data.npz')


if __name__ == "__main__":
    train_percentage = 0.8
    train_tensor, val_tensor = load_data(train_percentage)
    total_tensor = th.cat([train_tensor, val_tensor], axis=0)
    frequencies = estimate_token_frequency(total_tensor)
    print(frequencies)

    generate_dataset_from_original_levels()