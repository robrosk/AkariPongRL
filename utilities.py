import torch
import numpy as np

class Utilities:    
    @staticmethod
    def softmax(logits):
        """
        Compute softmax probabilities from logits using PyTorch.
        """
        return torch.nn.functional.softmax(logits, dim=-1)
    
    @staticmethod
    def normalize_expand_transpose_state(state):
        """
        Converts a NumPy state into a PyTorch tensor, normalizes, and transposes it.
        """
        state_tensor = torch.from_numpy(np.array(state)).float() / 255.0
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        return state_tensor
