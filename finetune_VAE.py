from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.level_generators import VAEGenerator
from src.utils.data_loading import generate_VAE_dataloader
from src.utils.load_models import load_VAE_model

import argparse

def run(**kwargs):
    wandb_logger = WandbLogger(
        project='mario_level_updating',
        log_model=True
    )   

    mario_train, mario_val, token_frequencies, frame_shape = \
        generate_VAE_dataloader(
            token_hidden=kwargs.get('token_hidden', 'q_mark'), 
            finetuning=True
        )
    kwargs['frame_shape'] = frame_shape
    kwargs['token_frequencies'] = token_frequencies
    mario_generator = load_VAE_model(**kwargs)

    callbacks = []
    if kwargs['use_early_stop']:
        callbacks.append(EarlyStopping(
            monitor="val_loss", min_delta=0.1, 
            patience=5, verbose=False, mode="min"
        ))
    # callbacks.append(
    #     ModelCheckpoint(monitor="val_loss", mode="min")
    # )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,          
        log_every_n_steps=4,
        max_epochs=240
    ) #(accelerator='gpu')
    trainer.fit(mario_generator, mario_train, mario_val)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_checkpoint', type=str)
    parser.add_argument('token_hidden', type=str, help='Token hidden in the dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='Latent dimension')
    parser.add_argument('--unet_detach_mode', type=str, default='features')
    parser.add_argument('-use_early_stop', action='store_true')
    args = parser.parse_args()

    run(**vars(args))
