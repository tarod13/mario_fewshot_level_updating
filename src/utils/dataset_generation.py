import os
import numpy as np
import torch as th
from torch.nn.functional import one_hot
from itertools import combinations, permutations
from copy import deepcopy
from collections import defaultdict
import logging
import argparse

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("GenerationLogger")


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

def unify_tensor(tensor_list):
    unified_tensor_list = th.cat(tensor_list, dim=1)\
        .transpose(0,1).transpose(1,2)
    return unified_tensor_list

def unify_save_tensor(tensor_list, file):
    unified_tensor_list = unify_tensor(tensor_list)
    print(unified_tensor_list.shape)
    np.savez(file, unified_tensor_list)

def find_possible_transformations(transformation_basis: list, r: int):
    possible_transformations = list(combinations(transformation_basis, r=r))
    possible_transformations = list(set([
        tuple(set(transform)) for transform in possible_transformations
    ]))
    possible_transformations.sort(key=len)
    return possible_transformations

def update_keys_of_dict(old_dict, transform_map):
    new_dict = old_dict.copy()
    for k, v in transform_map.items():
        if k in new_dict:
            new_dict[k] = old_dict[v]
    return new_dict

def generate_token_dicts(token_dicts, transformation_dict):
    transformation_pairs = []
    for transform_name, transform_map in transformation_dict.items():
        token_dicts_to_append = dict()
        for dict_name, token_dict in token_dicts.items():
            if 'original' in dict_name:
                new_dict_name = frozenset((transform_name,))
            else:
                new_dict_name = frozenset.union(dict_name, frozenset((transform_name,)))
            transformation_pairs.append((new_dict_name, dict_name))
            new_token_dict = update_keys_of_dict(token_dict, transform_map)
            token_dicts_to_append[new_dict_name] = new_token_dict
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

def generate_tensor_dict(
    path, token_dicts, **kwargs
):
    tensor_dict = defaultdict(list)
    for levelFile in os.listdir(path):
        # Log level being loaded
        logger.info("Processing: " + levelFile)   
        with open(path+ '/' + levelFile) as fp:
            level_txt = fp.read()
            list_rows = level_to_list(level_txt)

        for dict_name, token_dict in token_dicts.items():
            list_rows_copy = deepcopy(list_rows)            
            list_rows_tf = []
            level_tokenizer(list_rows_copy, list_rows_tf, token_dict) 
            level_np = level_to_array(list_rows_tf)
            narray_to_tensor(level_np, tensor_dict[dict_name], **kwargs)

    for dict_name, tensor_list in tensor_dict.items():
        game_tensor = unify_tensor(tensor_list)
        tensor_dict[dict_name] = game_tensor
    
    return tensor_dict

def generate_vae_dataset(
    tensor_dict, hold_out_settings, token_to_hide, 
    finetuning, train=True
):
    train_tensor_list = []
    test_tensor_list = []

    for setting, game_tensor in tensor_dict.items():
        if setting not in hold_out_settings:
            train_tensor_list.append(game_tensor)
        else:
            test_tensor_list.append(game_tensor)

    train_tensor = th.cat(train_tensor_list, dim=0)
    test_tensor = th.cat(test_tensor_list, dim=0)

    logger.info(f'Storing VAE train and test datasets...')
    logger.info(f'Train dataset size: {train_tensor.shape}')
    logger.info(f'Test dataset size: {test_tensor.shape}')
    th.save(train_tensor, f'./data/vae_dataset_{"train" if train else "test"}_no_{token_to_hide}{"" if not finetuning else "_finetuning"}.pt')
    # th.save(test_tensor, f'./data/vae_dataset_test_no_{token_to_hide}{"" if not finetuning else "_finetuning"}.pt')

def check_if_in_string(list_, string_):
    in_string = False
    for e in list_:
        if e in string_:
            in_string = True
            break
    return in_string        

def generate_transformation_nets_dataset(
    tensor_dict, transformation_pairs, hold_out_origins, 
    hold_out_destinations, token_to_hide, finetuning,
    train=True    
):
    train_tensor_list = []
    test_tensor_list = []

    for origin, destination in transformation_pairs:
        origin_tensor = tensor_dict[origin]
        destination_tensor = tensor_dict[destination]
        stacked_tensors = th.stack([origin_tensor, destination_tensor], dim=0)
        
        origin_hold_out = origin in hold_out_origins
        destination_hold_out = destination in hold_out_destinations
        if origin_hold_out or destination_hold_out:
            test_tensor_list.append(stacked_tensors)
        else:
            train_tensor_list.append(stacked_tensors)
            logger.info(f'Accepted for training: ({origin}, {destination})...')

    train_tensor = th.stack(train_tensor_list, dim=1)
    test_tensor = th.stack(test_tensor_list, dim=1)

    logger.info(f'Storing TransformationNet train and test datasets...')
    logger.info(f'Train dataset size: {train_tensor.shape}')
    logger.info(f'Test dataset size: {test_tensor.shape}')
    th.save(train_tensor, f'./data/tnet_dataset_{"train" if train else "test"}_no_{token_to_hide}{"" if not finetuning else "_finetuning"}.pt')
    # th.save(test_tensor, f'./data/tnet_dataset_test_no_{token_to_hide}{"" if not finetuning else "_finetuning"}.pt') 

def generate_permutations(list_):
    possible_permutations = list(permutations(list_))
    possible_permutations = ['-'.join(permutation_) for permutation_ in possible_permutations]
    return possible_permutations

def generate_dataset_from_original_levels(
    path: str = './data/basic_13_tokens',
    size: int = 14, 
    stride: int = 5, 
    num_classes: int = 13,
    token_to_hide: str = 'q_mark',
    finetuning: bool = False,
    train: bool = True,
):  
    # Definition of basic dictionaries
    param_dict = {
        'size': size,
        'stride': stride,
        'num_classes': num_classes,
    }
    
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
        'no_coin': {'o':'-'},
    }

    if not finetuning:
        if token_to_hide == 'q_mark':
            hold_out_settings = ['original','no_cannon','no_coin'] + generate_permutations(['no_coin','no_cannon'])            
        elif token_to_hide == 'cannon':
            hold_out_settings = ['original','no_q_mark','no_coin'] + generate_permutations(['no_coin','no_q_mark'])
        elif token_to_hide == 'coin':
            hold_out_settings = ['original','no_q_mark','no_cannon'] + generate_permutations(['no_cannon','no_q_mark'])
        else:
            raise ValueError("Invalid token")
        
        hold_out_origins = hold_out_settings.copy()
        hold_out_destinations = hold_out_settings.copy()
    else:
        if token_to_hide == 'q_mark':
            hold_out_origins = ['original','no_cannon','no_coin'] + generate_permutations(['no_coin','no_cannon'])
            hold_out_destinations = (
                ['no_q_mark']
                + generate_permutations(['no_coin','no_q_mark'])
                + generate_permutations(['no_q_mark','no_cannon'])
                + generate_permutations(['no_coin','no_cannon','no_q_mark'])
            )            
        elif token_to_hide == 'cannon':
            hold_out_origins = ['original','no_q_mark','no_coin'] + generate_permutations(['no_coin','no_q_mark'])
            hold_out_destinations = (
                ['no_cannon']
                + generate_permutations(['no_coin','no_cannon'])
                + generate_permutations(['no_q_mark','no_cannon'])
                + generate_permutations(['no_coin','no_cannon','no_q_mark'])
            )            
        elif token_to_hide == 'coin':
            hold_out_origins = ['original','no_q_mark','no_cannon'] + generate_permutations(['no_cannon','no_q_mark'])
            hold_out_destinations = (
                ['no_coin']
                + generate_permutations(['no_coin','no_q_mark'])
                + generate_permutations(['no_coin','no_cannon'])
                + generate_permutations(['no_coin','no_cannon','no_q_mark'])
            )
        else:
            raise ValueError("Invalid token")
        hold_out_settings = hold_out_destinations.copy()

        path = path + '_finetuning_' + token_to_hide + ('/finetuning' if train else '/evaluation')

    # Generation of token tensors 
    logger.info(f'Generating token dicts with transformations...')
    
    token_dicts, transformation_pairs = generate_token_dicts(
        token_dicts, transformation_dict
    )    
    transformation_path_dict = complete_transformation_pairs(
        transformation_pairs
    )
    token_dicts, transformation_pairs = simplify_token_dict_keys(
        token_dicts, transformation_path_dict
    )

    # Log generated token dicts
    for dict_name, token_dict in token_dicts.items():
        logger.info(f'Dict name: {dict_name}, Token dict: {token_dict}')

    logger.info(f'Generating game tensors for each transformation...')
    tensor_dict = generate_tensor_dict(path, token_dicts, **param_dict)

    # Generation of datasets
    logger.info(f'Generating dataset to train VAE...')
    generate_vae_dataset(
        tensor_dict, hold_out_settings, token_to_hide, finetuning, train
    )

    logger.info(f'Generating dataset to train VAE...')
    generate_transformation_nets_dataset(
        tensor_dict, transformation_pairs, hold_out_origins, 
        hold_out_destinations, token_to_hide, finetuning, train
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('token_to_hide', type=str)
    parser.add_argument('-finetune', action='store_true')
    parser.add_argument('-test', action='store_true')
    args = parser.parse_args()

    generate_dataset_from_original_levels(
        token_to_hide=args.token_to_hide, finetuning=args.finetune, train=(not args.test)
    )