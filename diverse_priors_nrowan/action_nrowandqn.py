import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from NoisyLinear import NoisyLinear
import numpy as np
   
"""
This NROWAN-DQN will be applied to Cartpole.
No need of convolution layers, state information is directly 
delivered to fully connected layer.
Fully connected layer has 2 hidden layers and 1 output layer.
Each hidden layer has 128 neurons,
Output layer has same number of neurons as number of environment actions.
All layers user ReLu function as activation function except output layer.

"""

class ACTION_NROWANDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, env, initial_noise=0.5, min_noise=0.01, decay_rate=0.995):
        super(ACTION_NROWANDQN, self).__init__()

        self.env = env
        self.num_actions=num_actions

        #fully connected layer with 2 hidden layers
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        # noisy layer
        self.noisy_fc3 = NoisyLinear(128, env.action_space.n)


        # adding noise parameters
        self.noise_scale = initial_noise
        self.min_noise = min_noise
        self.decay_rate = decay_rate

    # forward pass
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.noisy_fc3(x)
        return x

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to("cpu")  # Ensure state is a tensor and on the correct device

        # Add Gaussian action noise scaled by the noise scale
        # noise = np.random.normal(0, self.noise_scale, size=self.num_actions)

        # Get Q-values for all actions and add noise
        with torch.no_grad():
            # q_value = self.forward(state).cpu().data.numpy() + noise
            q_value = self.forward(state).cpu().data.numpy()

        action = np.argmax(q_value, axis=1).item()  # Select the action with the max Q-value and convert to Python scalar

        # Decay the noise scale
        # self.noise_scale = max(self.min_noise, self.noise_scale * self.decay_rate)
        # print("Noise Scale: ", str(self.noise_scale))

        return action

    def reset_noise(self):
        self.noisy_fc3.reset_noise()

    def get_sigmaloss(self):
        return self.noisy_fc3.sigma_loss()

if __name__ == '__main__':

    state_dim = 4
    action_dim = 2
    env = gym.make("CartPole-v1")
    net = ACTION_NROWANDQN(state_dim, action_dim, env)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
