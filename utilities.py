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
        Converts a NumPy state into a PyTorch tensor, normalizes appropriately.
        For image observations: normalizes by 255.0 and keeps shape (C, H, W)
        For vector observations: keeps as-is and flattens if needed
        """
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        
        state_tensor = torch.from_numpy(state).float()
        
        # Check if this is an image observation (3D with reasonable dimensions)
        if len(state.shape) >= 2 and state.shape[-1] >= 64 and state.shape[-2] >= 64:
            # Image observation - normalize by 255.0
            state_tensor = state_tensor / 255.0
        # For vector observations, keep values as-is (they should already be properly scaled)
        
        # Add batch dimension
        state_tensor = state_tensor.unsqueeze(0)
        return state_tensor
