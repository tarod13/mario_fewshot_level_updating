from typing import List

from torch import Tensor
from torch.optim import Optimizer
from pytorch_lightning import LightningModule

class BaseGenerator(LightningModule):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def configure_optimizers(self, optimizer_idx: int) -> Optimizer:
        raise NotImplementedError

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Tensor:
        raise NotImplementedError