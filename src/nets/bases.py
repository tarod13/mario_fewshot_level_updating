from typing import List, Any
from torch import nn
from torch import Tensor
from torch.distributions import Categorical
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Categorical:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class BaseTN(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Categorical:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass