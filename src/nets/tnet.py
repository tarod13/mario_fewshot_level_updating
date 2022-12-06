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
        '''Returns the approximate embedding of the transformed frame.'''
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
        '''Returns the categorical likelihood model p_X(.|e)'''
        p_x_given_embedding = self.unet.decode(embedding, skip_feature_list)
        return p_x_given_embedding

    def forward(self, XY: th.Tensor) -> List[Distribution]:
        '''
        Encodes and decodes the input, generating the approximate
        embedding e(x_transformed) and likelihood models p_X(.|e).
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
        up_val: float = 0.9
    ) -> th.Tensor:
        '''
        Calculates the MSE between the estimated and
        real embeddings.
        '''

        x = XY[:,0]
        x_transformed = XY[:,1]
        labels = x.argmax(dim=1)
        labels_transformed = x_transformed.argmax(dim=1)
        mask_diff = th.where(
            labels != labels_transformed, 
            th.ones_like(labels_transformed), 
            th.zeros_like(labels_transformed)
        ).float()
        with th.no_grad():
            x_transformed_embedding_real = self.encode_single(x_transformed)
        
        conditional_likelihood = p_x_given_embedding.log_prob(labels_transformed)
        update_loss_diff = -th.einsum(
            'hij,hij->h',
            conditional_likelihood,
            mask_diff
        ).mean()
        update_loss_same = -th.einsum(
            'hij,hij->h',
            conditional_likelihood,
            1-mask_diff
        ).mean()
        update_loss = up_val*update_loss_diff + (1-up_val)*update_loss_same

        embedding_difference = x_transformed_embedding_real - x_transformed_embedding
        embedding_loss = (embedding_difference**2).sum(1).mean()

        loss = 0.1*update_loss + embedding_loss
        return (
            loss, 
            update_loss_diff.detach().item(), 
            update_loss_same.detach().item(),
            embedding_loss.detach().item()
        )

    def sample_from_conditional_likelihood(
        self, p_x_given_embedding: Distribution 
    ) -> th.Tensor:
        '''Generate sample batch x from p_X(.|e).'''
        x = p_x_given_embedding.probs.argmax(dim=-1)
        return x

    def generate(
        self, XY: th.Tensor
    ) -> th.Tensor:
        '''Generates reconstruction of input.'''
        with th.no_grad():
            p_x_given_embedding = self(XY)[1]
        x = self.sample_from_conditional_likelihood(p_x_given_embedding)
        return x