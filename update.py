import os
import argparse
import wandb
from pathlib import Path

from src.utils.data_loading import load_pytorch_TN_dataset as load_data
from src.utils.load_models import load_TN_model
from src.utils.plotting import get_img_from_level


def get_checkpoint(vae_name, version):

    checkpoint_folder = f"./lightning_logs/version_{version}/checkpoints"
    save_files = os.listdir(checkpoint_folder)
    checkpoint = checkpoint_folder + '/' + save_files[0]
    return checkpoint, version

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_checkpoint', type=str)
    args = parser.parse_args()

    version = args.wandb_checkpoint.split("/")[-1].split(":",1)[0]
    mario_val = load_data(n_examples=1)[2]
    print(mario_val.shape)
    mario_generator = load_TN_model(**vars(args))    
    mario_generator.update(mario_val, version=version)


if __name__ == "__main__":
    run()
