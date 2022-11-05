from typing import List
import numpy as np
from torch import Tensor
from torch.optim import Optimizer, Adam

from nets.vaes import VanillaVAE
from level_generators import BaseGenerator


class VAEGenerator(BaseGenerator):
    def __init__(
        self, **kwargs
    ):
        super().__init__()
        self.VAE = VanillaVAE(**kwargs)
               

    def forward(self, input: Tensor) -> List[Tensor]:
        return self.VAE(input)

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(list(self.parameters()), lr=1e-3)   # TODO: set self.parameters
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

    # def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int):
    #     loss.backward()
