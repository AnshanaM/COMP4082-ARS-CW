import torch
from torch import nn
import torch.nn.functional as F
from NoisyLinear import NoisyLinear

# # standard way to define a network is by defining a class
# # this class inherits the nn.Module
# class DQN(nn.Module):
#     # parameters: dimension of input layer, dimension of output layer, dimension of hidden layer
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(DQN, self).__init__()

#         # for pytorch the input layer is implicit
#         # no need to write code for it

#         # defining the second layer: hidden layer
#         self.fc1 = nn.Linear(state_dim, hidden_dim)

#         # defining the output layer
#         self.fc2 = nn.Linear(hidden_dim, action_dim)

#     # parameter x in the input state, performing the activation function
#     # and then return to the output layer which calculates the q values
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    

class NROWANDQN(nn.Module):
    def __init__(self, state_dim, action_dim, sigma_init=0.017):
        super(NROWANDQN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.noisy_fc2 = NoisyLinear(128, 128, sigma_init=sigma_init)
        self.noisy_fc3 = NoisyLinear(128, action_dim, sigma_init=sigma_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))         # Standard FC layer
        x = F.relu(self.noisy_fc2(x))  # Noisy hidden layer
        return self.noisy_fc3(x)       # Noisy output layer

    def apply_gaussian_noise(self):
        # Apply Gaussian noise to the weights of all layers
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param.add_(torch.randn_like(param) * self.sigma_init)

    def adjust_weights(self, k_factor):
        # Adjust the weights dynamically
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(k_factor)


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = NROWANDQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)

