from src.level_generators import VAEGenerator
from src.utils.data import load_data
from src.utils.plotting import get_img_from_level

import os
import argparse


def get_checkpoint(vae_name, version):
    if vae_name == 'vanilla':
        version = 1        
    elif vae_name == 'unet':
        if not (version in [9, 12]):
            version = 12
    else:
        raise ValueError(f'Invalid VAE type: {vae_name}')

    checkpoint_folder = f"./lightning_logs/version_{version}/checkpoints"
    save_files = os.listdir(checkpoint_folder)
    checkpoint = checkpoint_folder + '/' + save_files[0]
    return checkpoint, version

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('vae', type=str, help='Name of VAE class to use')
    parser.add_argument('--version', type=int, help='Model version to load', required=False)
    args = parser.parse_args()

    checkpoint, version = get_checkpoint(args.vae, args.version)
    gen_dict = {'vae_name': args.vae}
    print(f'vae_name: {args.vae}, version: {version}')

    mario_val = load_data()[1]    
    
    mario_generator = VAEGenerator.load_from_checkpoint(checkpoint, **gen_dict)
    mario_generator.reconstruct(mario_val, version=version)


if __name__ == "__main__":
    run()
