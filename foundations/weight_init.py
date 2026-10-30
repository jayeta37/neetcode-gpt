import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)
        weights = []

        dimensions = [input_dim] + [hidden_dim] * num_layers    
        
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dimensions[i] + dimensions[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dimensions[i])
            else:
                std = 1.0
            weight = torch.randn(dimensions[i + 1], dimensions[i]) * std

            weights.append(weight)

        x = torch.randn(input_dim)
        stds = []

        for w in weights:
            x = torch.relu(w @ x)
            stds.append(round(x.std().item(), 2))
        return stds
