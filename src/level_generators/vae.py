from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torch.optim import Optimizer, Adam

from src.nets.vaes import VanillaVAE, UNetVAE
from src.level_generators import BaseGenerator
from src.utils.plotting import get_img_from_level


class VAEGenerator(BaseGenerator):
    def __init__(
        self, vae_name: str = 'vanilla', **kwargs, 
    ):
        super().__init__()
        if vae_name == 'vanilla':
            self.VAE = VanillaVAE(**kwargs)
        elif vae_name == 'unet':
            self.VAE = UNetVAE(**kwargs) 
        else:
            raise ValueError('Invalid VAE name: %s' % vae_name)

    def forward(self, input: Tensor) -> List[Tensor]:
        return self.VAE(input)

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(list(self.parameters()), lr=1e-4)   # TODO: set self.parameters
        return optimizer

    def step(self, x_batch: Tensor) -> Tensor:
        x_batch = x_batch[0]
        q_z_given_x, p_x_given_z = self.forward(x_batch)
        loss = self.VAE.loss_function(x_batch, q_z_given_x, p_x_given_z)
        return loss

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.step(train_batch)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, val_batch: Tensor, batch_idx: int) -> None:
        loss = self.step(val_batch)
        self.log("val_loss", loss)
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

        original = val_batch.argmax(dim=1)
        reconstruction = self.VAE.generate(val_batch)
        
        images_original = [
            get_img_from_level(original[i].cpu().detach().numpy()) 
            for i in range(val_batch.shape[0])
        ]

        images_reconstructed = [
            get_img_from_level(reconstruction[i].cpu().detach().numpy()) 
            for i in range(reconstruction.shape[0])
        ]

        padding = ((0,0),(0,boundary_size+2*pad),(0,0))
        images = np.array([
                np.concatenate((np.pad(o, padding), r), axis=1)
                for (o, r) in zip(images_original, images_reconstructed)
        ])
        
        bidx_i = images_original[0].shape[1] + pad
        bidx_f = bidx_i + boundary_size
        
        images[:,:,bidx_i-pad: bidx_i,:] = 255   # White before boundary
        images[:,:,bidx_f:bidx_f+pad,:] = 255    # White after boundary
        images[:,:,bidx_i:bidx_f,0] = 0.60*255   # R
        images[:,:,bidx_i:bidx_f,1] = 0.15*255   # G
        images[:,:,bidx_i:bidx_f,2] = 0.40*255   # B

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
