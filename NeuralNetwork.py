import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, action_space):
        super(NeuralNetwork, self).__init__()
        
        # Store action space info
        self.action_space = action_space
        self.is_discrete = hasattr(action_space, 'n')
        
        if self.is_discrete:
            self.num_actions = action_space.n
        else:
            # For continuous actions, we need mean and log_std for each action dimension
            self.num_actions = action_space.shape[0]
        
        # PyTorch expects input channels first: (N, C, H, W)
        # 1) Conv1: in_channels=4 -> out_channels=32, kernel=8x8, stride=4, ReLU
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4
        )
        
        # 2) Conv2: in_channels=32 -> out_channels=64, kernel=4x4, stride=2, ReLU
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )
        
        # 3) Conv3: in_channels=64 -> out_channels=64, kernel=3x3, stride=1, ReLU
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        
        # 4) Flatten layer is handled in the forward pass
        
        # 5) Dense: 3136 -> 512, ReLU
        self.dense = nn.Linear(7 * 7 * 64, 512)
        
        # 6) Actor head - different for discrete vs continuous
        if self.is_discrete:
            # Discrete actions: logits over actions
            self.logits = nn.Linear(512, self.num_actions)
        else:
            # Continuous actions: mean and log_std for each action dimension
            self.action_mean = nn.Linear(512, self.num_actions)
            self.action_log_std = nn.Parameter(torch.zeros(self.num_actions))
        
        # 7) Critic head (scalar value)
        self.value = nn.Linear(512, 1)

    def forward(self, inputs):
        """
        Forward pass:
          inputs: float32 tensor, shape = (batch_size, 4, 84, 84),
                  with pixel values normalized to [0,1].
        Returns:
          For discrete: (logits, value)
          For continuous: (action_mean, action_log_std, value)
        """
        # PyTorch's channel format is (N, C, H, W)
        # The Utilities.py file now correctly formats the state to (1, 4, 84, 84)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        h = F.relu(self.dense(x))
        
        # Value head (same for both)
        value = self.value(h)
        
        if self.is_discrete:
            # Discrete actions
            logits = self.logits(h)
            return logits, value
        else:
            # Continuous actions
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            return action_mean, action_log_std, value

# Example of running a dummy forward pass:
if __name__ == "__main__":
    print("Testing Neural Network...")
    # The input to the network is now channels-first (N, C, H, W)
    dummy_input = torch.rand(1, 4, 84, 84)
    
    # Test with discrete action space
    from gymnasium.spaces import Discrete
    discrete_action_space = Discrete(6)
    model_discrete = NeuralNetwork(discrete_action_space)
    output_discrete = model_discrete(dummy_input)
    
    print("Discrete Model architecture:")
    print(model_discrete)
    print("Discrete output shapes:")
    if len(output_discrete) == 2:
        logits_out, value_out = output_discrete
        print("Logits shape:", logits_out.shape)  # Expect: torch.Size([1, 6])
        print("Value shape: ", value_out.shape)   # Expect: torch.Size([1, 1])
    
    # Test with continuous action space
    from gymnasium.spaces import Box
    continuous_action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    model_continuous = NeuralNetwork(continuous_action_space)
    output_continuous = model_continuous(dummy_input)
    
    print("\nContinuous Model architecture:")
    print(model_continuous)
    print("Continuous output shapes:")
    if len(output_continuous) == 3:
        action_mean_out, action_log_std_out, value_out = output_continuous
        print("Action mean shape:", action_mean_out.shape)      # Expect: torch.Size([1, 2])
        print("Action log_std shape:", action_log_std_out.shape) # Expect: torch.Size([1, 2])
        print("Value shape:", value_out.shape)                  # Expect: torch.Size([1, 1])
