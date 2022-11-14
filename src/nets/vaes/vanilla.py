from typing import List

import numpy as np

import torch as th
import torch.nn as nn
from torch.distributions import (
    Distribution, Normal, Categorical, kl_divergence)

from src.nets.vaes import BaseVAE


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        c: int = 11,
        z_dim: int = 2,
        #device: str = None,
    ):
        super().__init__()
        self.w = w
        self.h = h
        self.c = c
        self.input_dim = w * h * c
        self.z_dim = z_dim
        #self.device = device or th.device("cuda" if th.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ) #.to(self.device)
        
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)) # TODO: remove unnecessary nn.Sequential
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim))# .to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ) #.to(self.device)

    def p_z(self, device):
        return Normal(
            th.zeros(self.z_dim, device=device),
            th.ones(self.z_dim, device=device)
        )

    def encode(self, x: th.Tensor) -> Normal:
        '''Returns the normal posterior model q_Z(.|x)'''
        x = x.view(-1, self.input_dim) #.to(self.device)
        r = self.encoder(x)
        mu = self.enc_mu(r)
        log_var = self.enc_var(r)
        std = th.exp(0.5*log_var)
        return Normal(mu, std)

    def decode(self, z: th.Tensor) -> Categorical:
        '''Returns the categorical likelihood model p_X(.|z)'''
        logits = self.decoder(z) #.to(self.device))
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.c)
        )
        return p_x_given_z

    def forward(self, x: th.Tensor) -> List[Distribution]:
        '''
        Encodes and decodes the input, generating the posterior 
        and likelihood models q_Z(.|x) and p_X(.|z).
        '''
        q_z_given_x = self.encode(x) #.to(self.device))
        z = q_z_given_x.rsample()
        p_x_given_z = self.decode(z) #.to(self.device))
        return [q_z_given_x, p_x_given_z]

    def loss_function(
        self, x: th.Tensor, q_z_given_x: Distribution, 
        p_x_given_z: Distribution,
    ) -> th.Tensor:
        '''
        Calculates the Evidence Lower BOund

        ELBO: = E_{q_Z(.|x)}[p_X(.|z)] - KL(q_Z(.|x)|p_Z)

        for the given input x, likelihood model p_X(.|z), 
        and posterior q_Z(.|x).
        '''
        labels = x.argmax(dim=1) # x.to(self.device).argmax(dim=1)
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
        z = self.p_z.sample(
            sample_shape=th.Size(num_samples, self.z_dim))
        with th.no_grad():
            p_x_given_z = self.decode(z) #.to(self.device))
        x = self.sample_from_conditional_likelihood(p_x_given_z)
        return x
    
    def generate(
        self, x: th.Tensor
    ) -> th.Tensor:
        '''Generates reconstruction of input.'''
        with th.no_grad():
            p_x_given_z = self(x)[1] #.to(self.device))[1]
        x = self.sample_from_conditional_likelihood(p_x_given_z)
        return x

