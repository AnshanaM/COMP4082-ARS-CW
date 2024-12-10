import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from NoisyLinear import NoisyLinear
import numpy as np
   
"""
This NROWAN-DQN will be applied to Cartpole and MountainCar
No need of convolution layers, state information is directly 
delivered to fully connected layer.
Fully connected layer has 2 hidden layers and 1 output layer.
Each hidden layer has 128 neurons,
Output layer has same number of neurons as number of environment actions.
All layers user ReLu function as activation function except output layer.

"""

class ORIGINAL_NROWANDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, env):
        super(ORIGINAL_NROWANDQN, self).__init__()

        self.env = env

        #fully connected layer with 2 hidden layers
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        # noisy layer
        self.noisy_fc3 = NoisyLinear(128, env.action_space.shape[0])

    # forward pass
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.noisy_fc3(x)
        return x
    
    def act(self,env,  state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to("cpu")
        with torch.no_grad():
            action = self.forward(state).squeeze(0)
            action = torch.tanh(action)  # Squash to [-1, 1]
            action = action * torch.FloatTensor(env.action_space.high)  # Scale to action bounds
            action = action.item()  # Convert to Python scalar       
        return np.array([action])

    
    def reset_noise(self):
        self.noisy_fc3.reset_noise()

    def get_sigmaloss(self):
        return self.noisy_fc3.sigma_loss()

if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    net = ORIGINAL_NROWANDQN(env.observation_space.shape[0], env.action_space.shape[0], env)
    state = torch.randn(1, env.observation_space.shape[0])
    output = net(state)
    print(output)
