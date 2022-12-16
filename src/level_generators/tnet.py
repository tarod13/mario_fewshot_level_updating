from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, randn
from torch.optim import Optimizer, Adam

from src.nets import TNet
from src.level_generators import BaseGenerator
from src.utils.training import create_mask
from src.utils.plotting import get_img_from_level, get_str_from_level

import pickle


def set_defaults(**kwargs):
    if 'z_dim' not in kwargs: kwargs['z_dim'] = 2
    if 'lr' not in kwargs: kwargs['lr'] = 1e-4
    return kwargs

class TNGenerator(BaseGenerator):
    def __init__(
        self, **kwargs, 
    ):
        super().__init__()
        
        kwargs = set_defaults(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.tnet = TNet(**kwargs)
        self.save_hyperparameters()
        self.use_mask = kwargs.get('use_mask', False)

    def forward(self, input: Tensor) -> List[Tensor]:
        return self.tnet(input)

    def configure_optimizers(self, **kwargs) -> Optimizer:
        if 'lr' not in kwargs:
            kwargs['lr'] = 1e-4
        optimizer = Adam(list(self.parameters()), lr=kwargs['lr'])   # TODO: set self.parameters
        return optimizer

    def step(self, XY_batch: Tensor) -> Tensor:
        XY_batch = XY_batch[0]
        if self.use_mask:
            x_batch = XY_batch[:,0]
            x_batch_masked, batch_mask = create_mask(
                x_batch, self.tnet.unet.token_frequencies
            )
            XY_batch[:,0] = x_batch_masked
        x_transformed_embedding, p_x_given_embedding = self.forward(XY_batch)
        loss = self.tnet.loss_function(
            x_transformed_embedding, XY_batch, p_x_given_embedding)
        return loss

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        loss, up_loss_diff, up_loss_same, emb_loss= self.step(train_batch)
        self.log("train_loss", loss)
        self.log("train_update_loss_diff", up_loss_diff)
        self.log("train_update_loss_same", up_loss_same)
        self.log("train_embedding_loss", emb_loss)
        return loss
        
    def validation_step(self, val_batch: Tensor, batch_idx: int) -> None:
        loss, up_loss_diff, up_loss_same, emb_loss = self.step(val_batch)
        self.log("val_loss", loss)
        self.log("val_update_loss_diff", up_loss_diff)
        self.log("val_update_loss_same", up_loss_same)
        self.log("val_embedding_loss", emb_loss)
        return loss

    def update(
        self, 
        val_batch: Tensor, 
        n_rows: int = 10, 
        n_cols: int = 4, 
        boundary_size: int = 4,
        pad: int = 30,
        frames_off: bool = False,
        version: int = 0,
        token_hidden: str = 'q_mark'
    ) -> None:

        indices = np.random.randint(0, high=val_batch.shape[0], size=n_rows*n_cols)
        batch = val_batch[indices]
        original = batch[:,0].argmax(dim=1)
        desired = batch[:,1].argmax(dim=1)
        original_ex = batch[:,2].argmax(dim=1)
        update_ex = batch[:,3].argmax(dim=1)
        update = self.tnet.generate(batch)
        
        strs_original = [
            get_str_from_level(original[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        strs_desired = [
            get_str_from_level(desired[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        strs_reconstructed = [
            get_str_from_level(update[i].cpu().detach().numpy()) 
            for i in range(update.shape[0])
        ]

        with open(f"data/desired_update_{token_hidden}.pickle", "wb") as output_file:
            pickle.dump(strs_desired, output_file)

        with open(f"data/update_{token_hidden}.pickle", "wb") as output_file:
            pickle.dump(strs_reconstructed, output_file)

        print(strs_original[0], strs_desired[0], strs_reconstructed[0])

        images_original = [
            get_img_from_level(original[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_desired = [
            get_img_from_level(desired[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_original_ex = [
            get_img_from_level(original_ex[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_update_ex = [
            get_img_from_level(update_ex[i].cpu().detach().numpy()) 
            for i in range(batch.shape[0])
        ]

        images_reconstructed = [
            get_img_from_level(update[i].cpu().detach().numpy()) 
            for i in range(update.shape[0])
        ]

        # padding = ((0,0),(0,boundary_size+2*pad),(0,0))
        # images = np.array([
        #         np.concatenate((np.pad(o, padding), r), axis=1)
        #         for (o, r) in zip(images_original, images_reconstructed)
        # ])
        
        # bidx_i = images_original[0].shape[1] + pad
        # bidx_f = bidx_i + boundary_size
        
        # images[:,:,bidx_i-pad: bidx_i,:] = 255   # White before boundary
        # images[:,:,bidx_f:bidx_f+pad,:] = 255    # White after boundary
        # images[:,:,bidx_i:bidx_f,0] = 0.60*255   # R
        # images[:,:,bidx_i:bidx_f,1] = 0.15*255   # G
        # images[:,:,bidx_i:bidx_f,2] = 0.40*255   # B

        padding = ((0,0),(0,boundary_size+2*pad),(0,0))
        images = np.array([
                np.concatenate([
                    np.pad(o, padding),
                    np.pad(d, padding),
                    np.pad(r, padding),
                    np.pad(oe, padding),
                    ue,
                ], axis=1)
                for (o, d, oe, ue, r) in zip(
                    images_original, 
                    images_desired,
                    images_original_ex,
                    images_update_ex, 
                    images_reconstructed)
        ])
        
        bidx_i = images_original[0].shape[1] + pad
        bidx_f = bidx_i + boundary_size
        
        for i in range(4):
            images[:,:,i*(bidx_f+pad)+bidx_i-pad: i*(bidx_f+pad)+bidx_i,:] = 255
            images[:,:,i*(bidx_f+pad)+bidx_f:(i+1)*(bidx_f+pad),:] = 255

            images[:,:,i*(bidx_f+pad)+bidx_i:i*(bidx_f+pad)+bidx_f,0] = 0.60*255   # R
            images[:,:,i*(bidx_f+pad)+bidx_i:i*(bidx_f+pad)+bidx_f,1] = 0.15*255   # G
            images[:,:,i*(bidx_f+pad)+bidx_i:i*(bidx_f+pad)+bidx_f,2] = 0.40*255   # B        
       
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
        plt.savefig(f'plots/updates_tnet_version_{version}.pdf', dpi=600)
        plt.show()
        plt.close()
