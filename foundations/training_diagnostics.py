import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        with torch.no_grad():
            for mod in model:
                x = mod(x)
                if isinstance(mod, nn.Linear):
                    result.append({
                        'mean' : x.mean().item(),
                        'std'  : x.std().item(),
                        'dead_fraction' : (x <= 0).all(dim=0).float().mean().item()
                    })

        return result

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_hat = model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        loss.backward()
        
        result = []
        for mod in model:
            if isinstance(mod, nn.Linear):
                result.append({
                    'mean'  : mod.weight.grad.mean().item(),
                    'std'   : mod.weight.grad.std().item(),
                    'norm'  : mod.weight.grad.norm().item()
                })

        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        for stats in activation_stats:
            if stats['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for grad_stats in gradient_stats:
            if grad_stats['norm'] > 1000:
                return 'exploding_gradients'
            elif grad_stats['norm'] < 1e-5:
                return 'vanishing_gradients'
            else:
                continue

        for stats in activation_stats:
            if stats['std'] < 0.1:
                return 'vanishing_gradients'
            elif stats['std'] > 10.0:
                return 'exploding_gradients'
            else:
                return 'healthy'