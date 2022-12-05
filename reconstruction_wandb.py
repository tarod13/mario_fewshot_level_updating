import os
import argparse
import wandb
from pathlib import Path

from src.level_generators import VAEGenerator
from src.utils.data_loading import load_pytorch_VAE_dataset as load_data
from src.utils.load_models import load_VAE_model
from src.utils.plotting import get_img_from_level


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
    parser.add_argument('wandb_checkpoint', type=str)
    args = parser.parse_args()

    version = args.wandb_checkpoint.split("/")[-1].split(":",1)[0]
    mario_val = load_data()[1]
    mario_generator = load_VAE_model(**vars(args))    
    mario_generator.reconstruct(mario_val, version=version)


if __name__ == "__main__":
    run()
