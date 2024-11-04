import torch
from torch import nn
import torch.nn.functional as F
from noisy_linear import NoisyLinear 


# standard way to define a network is by defining a class
# this class inherits the nn.Module
class DQN(nn.Module):
    # parameters: dimension of input layer, dimension of output layer, dimension of hidden layer
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        # for pytorch the input layer is implicit
        # no need to write code for it

        # defining the second layer: hidden layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # defining the output layer
        # self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.noisy_fc2 = NoisyLinear(hidden_dim, action_dim)

    # parameter x in the input state, performing the activation function
    # and then return to the output layer which calculates the q values
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # return self.fc2(x)
        return self.noisy_fc2(x)

    def reset_noise(self):
        self.noisy_fc2.reset_noise()

        
if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
