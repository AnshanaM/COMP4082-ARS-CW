import torch
from torch import nn
import torch.nn.functional as F

class NROWANDQN(nn.Module):
    def __init__(self, state_dim, action_dim, initial_sigma):
        super(NROWANDQN, self).__init__()

        # Initialize parameters for Gaussian noise
        self.sigma = initial_sigma  # Initial standard deviation for noise
        self.noise_scale = 1.0     # Scaling factor for the noise

        # Fully connected layers
        self.fc1 = nn.Linear(state_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, action_dim) # Output layer

    def forward(self, x):
        # Pass through fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output layer without activation

    def apply_gaussian_noise(self):
        # Apply Gaussian noise to the network parameters
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * self.sigma * self.noise_scale
                param.add_(noise)

    def adjust_weights(self, k_factor):
        # Online weight adjustment using a specified factor
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(k_factor)

if __name__ == '__main__':
    # Example state dimensions for CartPole
    state_dim = 4  # CartPole state has 4 features
    action_dim = 2 # CartPole has 2 actions (left, right)
    initial_sigma = 0.4

    net = NROWANDQN(state_dim, action_dim, initial_sigma)

    # Example input tensor (batch size, state_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
