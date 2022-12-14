from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, randn, ones_like, cat
from torch.optim import Optimizer, Adam

from src.nets import VanillaVAE, UNetVAE
from src.level_generators import BaseGenerator
from src.utils.training import create_mask
from src.utils.plotting import get_img_from_level


def set_defaults(**kwargs):
    if 'vae_name' not in kwargs: kwargs['vae_name'] = 'vanilla'
    if 'z_dim' not in kwargs: kwargs['z_dim'] = 2
    if 'lr' not in kwargs: kwargs['lr'] = 1e-4
    return kwargs

class VAEGenerator(BaseGenerator):
    def __init__(
        self, **kwargs, 
    ):
        super().__init__()
        
        kwargs = set_defaults(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if kwargs['vae_name'] == 'vanilla':
            self.VAE = VanillaVAE(**kwargs)
        elif kwargs['vae_name'] == 'unet':
            self.VAE = UNetVAE(**kwargs)
        else:
            raise ValueError('Invalid VAE name: %s' % kwargs['vae_name'])

        self.save_hyperparameters()

    def forward(self, input: Tensor) -> List[Tensor]:
        return self.VAE(input)

    def configure_optimizers(self, **kwargs) -> Optimizer:
        if 'lr' not in kwargs:
            kwargs['lr'] = 1e-4
        optimizer = Adam(list(self.parameters()), lr=kwargs['lr'])   # TODO: set self.parameters
        return optimizer

    def step(self, x_batch: Tensor) -> Tensor:
        x_batch = x_batch[0]
        x_batch_masked, batch_mask = create_mask(
            x_batch, self.VAE.token_frequencies
        )
        q_z_given_x, p_x_given_z = self.forward(x_batch_masked)
        loss, rec_loss, imp_loss = self.VAE.loss_function(
            x_batch, q_z_given_x, p_x_given_z, x_batch_masked, batch_mask)
        return loss, rec_loss, imp_loss

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        loss, rec_loss, imp_loss = self.step(train_batch)
        self.log("train_loss", loss)
        self.log("train_reconstruction_loss", rec_loss)
        self.log("train_impainting_loss", imp_loss)
        return loss
        
    def validation_step(self, val_batch: Tensor, batch_idx: int) -> None:
        loss, rec_loss, imp_loss = self.step(val_batch)
        self.log("val_loss", loss)
        self.log("val_reconstruction_loss", rec_loss)
        self.log("val_impainting_loss", imp_loss)
        return loss

    def reconstruct(
        self, 
        val_batch: Tensor, 
        n_rows: int = 5, 
        n_cols: int = 3, 
        boundary_size: int = 4,
        pad: int = 30,
        frames_off: bool = False,
        version: int = 0,
    ) -> None:

        indices = np.random.randint(0, high=val_batch.shape[0], size=n_rows*n_cols)
        batch = val_batch[indices]
        batch_masked = create_mask(
            batch, self.VAE.token_frequencies
        )[0]
        original = batch.argmax(dim=1)
        masked = cat([batch_masked, ones_like(batch_masked[:,0:1])], dim=1)\
            .argmax(dim=1)
        reconstruction = self.VAE.generate(batch_masked)

        # noise = 0*randn(batch.shape)
        # original = (batch+noise).argmax(dim=1)
        # reconstruction = self.VAE.generate(batch+noise)
        
        images_original = [
            get_img_from_level(original[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_masked = [
            get_img_from_level(masked[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_reconstructed = [
            get_img_from_level(reconstruction[i].cpu().detach().numpy()) 
            for i in range(reconstruction.shape[0])
        ]

        padding = ((0,0),(0,boundary_size+2*pad),(0,0))
        images = np.array([
                np.concatenate([
                    np.pad(o, padding),
                    np.pad(m, padding),
                    r,
                ], axis=1)
                for (o, m, r) in zip(
                    images_original, images_masked, images_reconstructed)
        ])
        
        bidx_i = images_original[0].shape[1] + pad
        bidx_f = bidx_i + boundary_size
        
        images[:,:,bidx_i-pad: bidx_i,:] = 255   # White before boundary
        images[:,:,bidx_f:bidx_f+pad,:] = 255    # White after boundary

        images[:,:,bidx_f+pad+bidx_i-pad: bidx_f+pad+bidx_i,:] = 255
        images[:,:,2*bidx_f+pad:2*bidx_f+2*pad,:] = 255

        images[:,:,bidx_i:bidx_f,0] = 0.60*255   # R
        images[:,:,bidx_i:bidx_f,1] = 0.15*255   # G
        images[:,:,bidx_i:bidx_f,2] = 0.40*255   # B

        images[:,:,bidx_f+pad+bidx_i:2*bidx_f+pad,0] = 0.60*255   # R
        images[:,:,bidx_f+pad+bidx_i:2*bidx_f+pad,1] = 0.15*255   # G
        images[:,:,bidx_f+pad+bidx_i:2*bidx_f+pad,2] = 0.40*255   # B        

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,10))
        for r in range(0, n_rows):
            for c in range(0, n_cols):
                idx = r*n_cols + c
                ax[r, c].imshow(255 * np.ones_like(images[idx]))  # White background
                ax[r, c].imshow(images[idx])
                ax[r, c].set_aspect('equal')
                if frames_off:
                    ax[r, c].axis("off")
                else:
                    ax[r, c].set_xticks([])
                    ax[r, c].set_yticks([])

        plt.tight_layout()             
        plt.savefig(f'plots/reconstructions_vae_version_{version}.pdf', dpi=600)
        plt.show()
        plt.close()

    # def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int):
    #     loss.backward()
