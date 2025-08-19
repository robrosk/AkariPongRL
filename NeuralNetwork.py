import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, action_space, observation_space):
        super(NeuralNetwork, self).__init__()
        
        # Store action space info
        self.action_space = action_space
        self.observation_space = observation_space
        self.is_discrete = hasattr(action_space, 'n')
        
        if self.is_discrete:
            self.num_actions = action_space.n
        else:
            # For continuous actions, we need mean and log_std for each action dimension
            self.num_actions = action_space.shape[0]
        
        # Determine if we're dealing with image observations or vector observations
        obs_shape = observation_space.shape
        self.is_image_obs = len(obs_shape) >= 3 and obs_shape[-1] >= 64 and obs_shape[-2] >= 64
        
        if self.is_image_obs:
            # Image observations: Use CNN architecture
            # PyTorch expects input channels first: (N, C, H, W)
            
            # 1) Conv1: in_channels=4 -> out_channels=32, kernel=8x8, stride=4, ReLU
            self.conv1 = nn.Conv2d(
                in_channels=obs_shape[0] if len(obs_shape) == 3 else 4,
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
            
            # Calculate the flattened size after convolutions dynamically
            # This works for any input size
            with torch.no_grad():
                dummy_input = torch.zeros(1, obs_shape[0] if len(obs_shape) == 3 else 4, 
                                        obs_shape[-2], obs_shape[-1])
                dummy_conv = self.conv3(self.conv2(self.conv1(dummy_input)))
                self.conv_output_size = dummy_conv.numel() // dummy_conv.size(0)
            
            # Multiple dense layers after CNN for better feature processing
            self.cnn_fc1 = nn.Linear(self.conv_output_size, 1024)
            self.cnn_fc2 = nn.Linear(1024, 512)
            
        else:
            # Vector observations: Use fully-connected architecture
            input_size = np.prod(obs_shape)  # Flatten the observation space
            
            # Multi-layer fully-connected network
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 512)
        
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
          inputs: float32 tensor, shape depends on observation type:
                  - For image obs: (batch_size, channels, height, width) with pixel values normalized to [0,1]
                  - For vector obs: (batch_size, obs_dim) with normalized values
        Returns:
          For discrete: (logits, value)
          For continuous: (action_mean, action_log_std, value)
        """
        if self.is_image_obs:
            # CNN path for image observations
            # PyTorch's channel format is (N, C, H, W)
            x = F.relu(self.conv1(inputs))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            # Flatten the output for the dense layers
            x = x.view(x.size(0), -1)
            
            # Multiple fully-connected layers for better feature processing
            x = F.relu(self.cnn_fc1(x))
            x = F.dropout(x, p=0.2, training=self.training)  # Add dropout for regularization
            h = F.relu(self.cnn_fc2(x))
        else:
            # Fully-connected path for vector observations
            # Flatten input if needed
            x = inputs.view(inputs.size(0), -1)
            
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.1, training=self.training)  # Light dropout for regularization
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.1, training=self.training)
            h = F.relu(self.fc3(x))
        
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
    from gymnasium.spaces import Discrete, Box
    
    # Test with image observations (CNN path)
    print("=== Testing Image Observations (CNN) ===")
    image_obs_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
    dummy_image_input = torch.rand(1, 4, 84, 84)
    
    # Test discrete actions with image observations
    discrete_action_space = Discrete(6)
    model_discrete_image = NeuralNetwork(discrete_action_space, image_obs_space)
    output_discrete_image = model_discrete_image(dummy_image_input)
    
    print("Discrete + Image Model architecture:")
    print(model_discrete_image)
    print("Discrete + Image output shapes:")
    if len(output_discrete_image) == 2:
        logits_out, value_out = output_discrete_image
        print("Logits shape:", logits_out.shape)  # Expect: torch.Size([1, 6])
        print("Value shape: ", value_out.shape)   # Expect: torch.Size([1, 1])
    
    # Test with vector observations (FC path)
    print("\n=== Testing Vector Observations (Fully-Connected) ===")
    vector_obs_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    dummy_vector_input = torch.rand(1, 4)
    
    # Test discrete actions with vector observations
    model_discrete_vector = NeuralNetwork(discrete_action_space, vector_obs_space)
    output_discrete_vector = model_discrete_vector(dummy_vector_input)
    
    print("Discrete + Vector Model architecture:")
    print(model_discrete_vector)
    print("Discrete + Vector output shapes:")
    if len(output_discrete_vector) == 2:
        logits_out, value_out = output_discrete_vector
        print("Logits shape:", logits_out.shape)  # Expect: torch.Size([1, 6])
        print("Value shape: ", value_out.shape)   # Expect: torch.Size([1, 1])
    
    # Test continuous actions with vector observations
    print("\n=== Testing Continuous Actions + Vector Observations ===")
    continuous_action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)
    model_continuous_vector = NeuralNetwork(continuous_action_space, vector_obs_space)
    output_continuous_vector = model_continuous_vector(dummy_vector_input)
    
    print("Continuous + Vector Model architecture:")
    print(model_continuous_vector)
    print("Continuous + Vector output shapes:")
    if len(output_continuous_vector) == 3:
        action_mean_out, action_log_std_out, value_out = output_continuous_vector
        print("Action mean shape:", action_mean_out.shape)      # Expect: torch.Size([1, 2])
        print("Action log_std shape:", action_log_std_out.shape) # Expect: torch.Size([1, 2])
        print("Value shape:", value_out.shape)                  # Expect: torch.Size([1, 1])
