from typing import List

import numpy as np

import torch as th
import torch.nn as nn
from torch.distributions import (
    Distribution, Normal, Categorical, kl_divergence)

from src.nets.vaes import BaseVAE


def calculate_ConvTranspose2d_output_size(
    in_size, kernel_size, stride, output_padding):
    if not isinstance(in_size, list):
        in_size = [in_size]
    out_size = [
        (in_size[i]-1)*stride + kernel_size[i] + output_padding
        for i in range(len(in_size))
    ]
    if len(out_size) == 1:
        out_size = 2*out_size
    return out_size

class Conv2dBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        in_size,
    ):
        super().__init__()
        self.feature_layer = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=1, 
            padding='same', 
        )
        self.norm_layer = nn.LayerNorm([out_channels, in_size[0], in_size[1]])
        self.activation_layer = nn.LeakyReLU()
        self.downscale_layer = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=kernel_size, stride=stride,
        )
    
    def forward(self, x):
        input, skip_feature_list = x
        features = self.norm_layer(self.feature_layer(input))
        output = self.downscale_layer(self.activation_layer(features))
        skip_feature_list.append(features)
        return [output, skip_feature_list]

class ConvTranspose2dBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size, 
        stride, 
        output_padding,
    ):
        super().__init__()

        self.upscale_layer = nn.ConvTranspose2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, 
            output_padding=output_padding
        )        
        self.reconstruction_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding='same',
        )
    
    def forward(self, x):
        input, skip_feature_list = x
        delta = self.upscale_layer(input)
        features = skip_feature_list[-1].detach()
        output = self.reconstruction_layer(features+delta)
        if len(skip_feature_list) > 1:
            return [output, skip_feature_list[:-1]]
        else:
            return output

class ActivationBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        size,
    ):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.LayerNorm([channels, size[0], size[1]]),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        input, list_ = x
        output = self.pipe(input)
        return [output, list_]

class UNetVAE(BaseVAE):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        c: int = 11,
        z_dim: int = 2,
    ):
        super().__init__()
        self.w = w
        self.h = h
        self.c = c
        self.input_dim = w * h * c
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            Conv2dBlock( c,  32, (3,3), 2, (h,w)),   # output: ( 32,6,6)
            ActivationBlock( 32, (6,6)),
            Conv2dBlock(32,  64, (3,3), 2, (6,6)),   # output: ( 64,2,2) 
            ActivationBlock( 64, (2,2)),
            Conv2dBlock(64, 128, (2,2), 1, (2,2)),   # output: (128,1,1) 
            ActivationBlock(128, (1,1)),  
        )
        self.enc_mu = nn.Linear(128, z_dim)
        self.enc_var = nn.Linear(128, z_dim)

        self.decoder_input = nn.Linear(z_dim, 128)
        self.decoder = nn.Sequential(
            ConvTranspose2dBlock(128, 64, (2,2), 1, 0),   # output: (64,2,2) 
            ActivationBlock(64, (2,2)),
            ConvTranspose2dBlock( 64, 32, (3,3), 2, 1),   # output: (32,6,6) 
            ActivationBlock(32, (6,6)),
            ConvTranspose2dBlock( 32,  c, (3,3), 2, 1),   # output: ( c,h,w)
        )

    def p_z(self, device):
        return Normal(
            th.zeros(self.z_dim, device=device),
            th.ones(self.z_dim, device=device)
        )

    def encode(self, x: th.Tensor) -> Normal:
        '''Returns the normal posterior model q_Z(.|x)'''
        r, skip_feature_list = self.encoder([x,[]])
        r_flat = r.view(-1, 128)
        mu = self.enc_mu(r_flat)
        log_var = self.enc_var(r_flat)
        std = th.exp(0.5*log_var)
        return [Normal(mu, std), skip_feature_list]

    def decode(self, z: th.Tensor, skip_feature_list: th.Tensor) -> Categorical:
        '''Returns the categorical likelihood model p_X(.|z)'''
        input_flat = self.decoder_input(z)
        input = input_flat.reshape(tuple(input_flat.shape)+(1,1))
        logits = self.decoder([input, skip_feature_list])
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.c)
        )
        return p_x_given_z

    def forward(self, x: th.Tensor) -> List[Distribution]:
        '''
        Encodes and decodes the input, generating the posterior 
        and likelihood models q_Z(.|x) and p_X(.|z).
        '''
        q_z_given_x, skip_feature_list = self.encode(x) 
        z = q_z_given_x.rsample()
        p_x_given_z = self.decode(z, skip_feature_list) 
        return [q_z_given_x, p_x_given_z]

    def loss_function(
        self, x: th.Tensor, q_z_given_x: Distribution, 
        p_x_given_z: Distribution
    ) -> th.Tensor:
        '''
        Calculates the Evidence Lower BOund

        ELBO: = E_{q_Z(.|x)}[p_X(.|z)] - KL(q_Z(.|x)|p_Z)

        for the given input x, likelihood model p_X(.|z), 
        and posterior q_Z(.|x).
        '''
        labels = x.argmax(dim=1) 
        conditional_likelihood = p_x_given_z.log_prob(labels).sum(dim=(1, 2))
        latent_divergence = kl_divergence(
            q_z_given_x, self.p_z(labels.device)
        ).sum(dim=1)
        elbo = conditional_likelihood - latent_divergence
        elbo_loss = (-elbo).mean()
        return elbo_loss    
    
    def sample_from_conditional_likelihood(
        self, p_x_given_z: Distribution 
    ) -> th.Tensor:
        '''Generate sample batch x from p_X(.|Z).'''
        x = p_x_given_z.probs.argmax(dim=-1)
        return x

    def sample(
        self, num_samples: int
    ) -> th.Tensor:
        '''
        Generates samples that follow the marginal 
        distribution p(x) = \int p_X(x|z)p(z).
        '''
        print('Cannot generate without input...')
        pass
    
    def generate(
        self, x: th.Tensor
    ) -> th.Tensor:
        '''Generates reconstruction of input.'''
        with th.no_grad():
            p_x_given_z = self(x)[1] 
        x = self.sample_from_conditional_likelihood(p_x_given_z)
        return x

