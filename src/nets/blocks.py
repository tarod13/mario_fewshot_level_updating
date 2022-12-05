import numpy as np

import torch as th
import torch.nn as nn


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
        **kwargs
    ):
        super().__init__()

        if 'unet_detach_mode' in kwargs:
            self.detach_mode = kwargs['unet_detach_mode']
        else:
            self.detach_mode = 'features'

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
        features = skip_feature_list[-1]
        
        if self.detach_mode == 'features':
            features = features.detach()
        elif self.detach_mode == 'delta':
            delta = delta.detach()

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