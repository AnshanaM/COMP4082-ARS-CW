import torch
from torch import nn
import torch.nn.functional as F
from NoisyLinear import NoisyLinear
   
#     def apply_gaussian_noise(self):
#         # Apply Gaussian noise to the weights of all layers
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 if 'weight' in name or 'bias' in name:
#                     param.add_(torch.randn_like(param) * self.sigma_init)

class NROWANDQN(nn.Module):
    def __init__(self, state_dim, action_dim, initial_delta):
        super(NROWANDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.noisy_fc3 = NoisyLinear(128, action_dim, initial_delta)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.noisy_fc3(x)

    def apply_gaussian_noise(self):
        # Explicitly trigger re-sampling of noise
        pass  # No action required; noise is generated in the NoisyLinear forward pass


    def adjust_weights(self, k_factor):
        # Adjust the weights dynamically
        with torch.no_grad():
            for param in self.parameters():
                param.mul_(k_factor)


if __name__ == '__main__':
    state_dim = 4
    action_dim = 2
    net = NROWANDQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)

