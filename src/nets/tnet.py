from typing import List

import numpy as np

import torch as th
import torch.nn as nn
from torch.distributions import (
    Distribution, Normal, Categorical, kl_divergence)

from src.nets import BaseTN, UNetVAE
from src.nets.blocks import Conv2dBlock, ConvTranspose2dBlock, ActivationBlock


class TNet(BaseTN):
    def __init__(
        self,
        frame_shape,
        z_dim: int = 2,
        token_frequencies: th.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.unet = UNetVAE(frame_shape, z_dim, token_frequencies, **kwargs)
        self.z_dim = z_dim

        self.transformation_embedding_net = nn.Sequential(
            nn.Linear(2*z_dim, 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, z_dim),
        )

        self.transformation_application_net = nn.Sequential(
            nn.Linear(2*z_dim, 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, z_dim),
        )

    def encode_single(self, x: th.Tensor) -> th.Tensor:
        x_embedding = self.unet.encode(x, test=True)[0]
        return x_embedding

    def encode(self, XY: th.Tensor) -> List:
        '''Returns the normal posterior model q_Z(.|x)'''
        x = XY[:,0]
        y = XY[:,2]
        y_transformed = XY[:,3]

        with th.no_grad():
            x_embedding, x_skip_feature_list = self.unet.encode(x, test=True)
            y_embedding = self.encode_single(y)
            y_transformed_embedding = self.encode_single(y_transformed)

        transformation_embedding_input = th.cat(
            [y_embedding, y_transformed_embedding], dim=1)
        transformation_embedding = self.transformation_embedding_net(
            transformation_embedding_input)
        norm_ = th.norm(transformation_embedding, dim=1, keepdim=True)
        transformation_embedding_normalized = (
            transformation_embedding / (norm_ + 1e-6))

        transformation_application_input = th.cat(
            [x_embedding, transformation_embedding_normalized], dim=1)
        x_transformed_embedding = self.transformation_application_net(
            transformation_application_input)

        return [x_transformed_embedding, x_skip_feature_list]

    def decode(
        self, embedding: th.Tensor, 
        skip_feature_list: th.Tensor
    ) -> Categorical:
        '''Returns the categorical likelihood model p_X(.|z)'''
        p_x_given_embedding = self.unet.decode(embedding, skip_feature_list)
        return p_x_given_embedding

    def forward(self, XY: th.Tensor) -> List[Distribution]:
        '''
        Encodes and decodes the input, generating the posterior 
        and likelihood models q_Z(.|x) and p_X(.|z).
        '''
        x_transformed_embedding, skip_feature_list = self.encode(XY) 
        p_x_given_embedding = self.decode(
            x_transformed_embedding, skip_feature_list
        ) 
        return [x_transformed_embedding, p_x_given_embedding]

    def loss_function(
        self, 
        x_transformed_embedding: th.Tensor, 
        XY: th.Tensor, 
        p_x_given_embedding: Distribution,
    ) -> th.Tensor:
        '''
        Calculates the MSE between the estimated and
        real embeddings.
        '''
        
        x_transformed = XY[:,1]
        with th.no_grad():
            x_transformed_embedding_real = self.encode_single(x_transformed)
        
        embedding_difference = x_transformed_embedding_real - x_transformed_embedding
        loss = (embedding_difference**2).sum(1).mean()
        return loss
