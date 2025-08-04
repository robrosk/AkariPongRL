import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super(NeuralNetwork, self).__init__()
        
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
        
        # 6) Actor head (logits over actions)
        self.logits = nn.Linear(512, num_actions)
        
        # 7) Critic head (scalar value)
        self.value = nn.Linear(512, 1)

    def forward(self, inputs):
        """
        Forward pass:
          inputs: float32 tensor, shape = (batch_size, 4, 84, 84),
                  with pixel values normalized to [0,1].
        Returns:
          logits: (batch_size, num_actions)
          value:  (batch_size, 1)
        """
        # PyTorch's channel format is (N, C, H, W)
        # The Utilities.py file now correctly formats the state to (1, 4, 84, 84)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        h = F.relu(self.dense(x))
        
        logits = self.logits(h)
        value = self.value(h)
        
        return logits, value

# Example of running a dummy forward pass:
if __name__ == "__main__":
    print("Testing Neural Network...")
    # The input to the network is now channels-first (N, C, H, W)
    dummy_input = torch.rand(1, 4, 84, 84)
    model = NeuralNetwork(num_actions=6)
    logits_out, value_out = model(dummy_input)
    
    print("Model architecture:")
    print(model)
    
    print("Logits shape:", logits_out.shape)  # Expect: torch.Size([1, 6])
    print("Value shape: ", value_out.shape)   # Expect: torch.Size([1, 1])
