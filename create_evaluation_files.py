# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:34:22 2022

@author: adeel
"""

import pickle
import os
from pathlib import Path


def create_txt_from_pickle(token_hidden):
    pickle_path: str = fr'./data/update_{token_hidden}.pickle'
    infile = open(pickle_path,'rb')
    update_dict = pickle.load(infile)    

    Path(fr"./data/Update/{token_hidden}").mkdir(parents=True, exist_ok=True)
    Path(fr"./data/Desired Update/{token_hidden}").mkdir(parents=True, exist_ok=True)

    n=0
    for i in update_dict:
        n = n+1
        with open(fr'./data/Update/{token_hidden}/file'+str(n)+'.txt', 'w') as f:
            for line in i:
                f.write(f"{line}")

    pickle_path: str = fr'./data/desired_update_{token_hidden}.pickle'
    
    infile = open(pickle_path,'rb')
    desired_update = pickle.load(infile)

    n=0
    for i in desired_update:
        n = n+1
        with open(fr'./data/Desired Update/{token_hidden}/file'+str(n)+'.txt', 'w') as f:
            for line in i:
                f.write(f"{line}")


if __name__ == "__main__":
    
    tokens = ['q_mark', 'cannon', 'coin']
    for token in tokens:
        create_txt_from_pickle(token)
    