from src.level_generators import VAEGenerator
from src.utils.data import load_data
from src.utils.plotting import get_img_from_level

import os
import argparse
import wandb
from pathlib import Path


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
    #parser.add_argument('vae_name', type=str, help='Name of VAE class to use')
    #parser.add_argument('--z_dim', type=int, default=2, help='Latent dimension')
    args = parser.parse_args()

    mario_val = load_data()[1]

    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    # checkpoint_reference = "tarod13/mario_level_updating/model-2afjdvgc:v0"
    # checkpoint_reference = "tarod13/mario_level_updating/model-crn52vpd:v0"   
    #checkpoint_reference = "tarod13/mario_level_updating/model-3d53z202:v0"     
    # version = checkpoint_reference.split(":",1)[1]
    version = args.wandb_checkpoint.split("/")[-1].split(":",1)[0]

    # download checkpoint locally (if not already cached)
    run = wandb.init(project="mario_level_updating")
    artifact = run.use_artifact(args.wandb_checkpoint, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    mario_generator = VAEGenerator.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt")
    mario_generator.reconstruct(mario_val, version=version)


if __name__ == "__main__":
    run()
