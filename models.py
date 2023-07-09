from typing import List, Tuple, Dict, Union, Any, cast, Optional
import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function

from efficientnet_pytorch import EfficientNet

class AutoencoderEfficientnet(nn.Module):
    def __init__(self, output_channels=3):
        super(AutoencoderEfficientnet, self).__init__()

        # Use EfficientNet-b0 pretrained on ImageNet as encoder
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')

        # Replace the classification head with a decoding head
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(1280, 512, 2, stride=2),  # upscale to 1/16 of original
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 2, stride=2),  # upscale to 1/8 of original
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 2, stride=2),  # upscale to 1/4 of original
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 2, stride=2),  # upscale to 1/2 of original
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 2, stride=2),  # upscale to original size
        nn.ReLU(True),
        nn.Conv2d(32, output_channels, 3, padding=1),  # Final convolution to get the right number of channels
        nn.Sigmoid()  # Activation function to get the values in the right range
        )


    def forward(self, x):
        with torch.no_grad():  # no need to calculate gradients for encoder
            features = self.encoder.extract_features(x)
        reconstruction = self.decoder(features)
        return reconstruction


# Define a simple Autoencoder
class Autoencoder(nn.Module):
    def __init__(self,input_size=128*128*3,hidden_size=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True))
        
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, input_size),
            nn.Tanh())
        
        n_nodes = 3
        self.nodes: List[Node] = [Node(memory_dim=hidden_size) for _ in range(n_nodes)]
        self.n_nodes: int = len(self.nodes)
        
        # self.response: torch.Tensor = torch.zeros(hidden_size)

    def forward(self, x, delta: float = 0.1, node_idx: int = 0):
        x = self.encoder(x)
        # self.response[:] = [node(x) for node in self.nodes]
        # Populate self.response efficiently
        response = delta*self.nodes[0](x)
        for node in self.nodes[1:]:
            response += delta*node(x)
        x = (1-self.n_nodes)*x + response
        x = self.decoder(x)
        
        return x
    
class ThresholdFun(Function):
    """
    Adaptive treshold for estimating node inputs in the memory module.
    """
    
    @staticmethod
    def forward(ctx, input, treshold):
        ctx.save_for_backward(input, treshold)
        return input.gt(treshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, treshold = ctx.saved_tensors
        grad_input = grad_treshold = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input.le(treshold)] = 0
            
        if ctx.needs_input_grad[1]:
            grad_treshold = grad_output.clone()
            grad_treshold[input.gt(treshold)] = 0

        return grad_input, grad_treshold
    
class Treshold(nn.Module):
    def __init__(self, treshold: float = 0.5):
        super(Treshold, self).__init__()
        self.treshold = torch.tensor(treshold, requires_grad=True)
        
    def forward(self, input):
        return ThresholdFun.apply(input, self.treshold)
    
class Node(nn.Module):
    def __init__(self, memory_dim: int = 128):
        super(Node, self).__init__()
        self.memory_dim = memory_dim
        self.weights: torch.Tensor = torch.randn(memory_dim, requires_grad=True)

        self.treshold = Treshold(treshold=0.5)

        
    def forward(self, x: torch.Tensor):

        response = x @ self.weights.T
        response = torch.nn.functional.relu(self.treshold(response))
            
        return response*self.weights

