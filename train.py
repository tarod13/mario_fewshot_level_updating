from pytorch_lightning import Trainer

from src.level_generators import VAEGenerator
from src.utils.data import generate_dataset

import argparse

def run(vae_name: str = 'vanilla'):
    mario_train, mario_val = generate_dataset()
    mario_generator = VAEGenerator(vae_name)
    trainer = Trainer() #(accelerator='gpu')
    trainer.fit(mario_generator, mario_train, mario_val)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('vae', type=str, help='Name of VAE class to use')
    args = parser.parse_args()

    run(vae_name = args.vae)
