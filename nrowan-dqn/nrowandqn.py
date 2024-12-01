import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from NoisyLinear import NoisyLinear
   
"""
This NROWAN-DQN will be applied to Cartpole and MountainCar
No need of convolution layers, state information is directly 
delivered to fully connected layer.
Fully connected layer has 2 hidden layers and 1 output layer.
Each hidden layer has 128 neurons,
Output layer has same number of neurons as number of environment actions.
All layers user ReLu function as activation function except output layer.

"""

class NROWANDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, env):
        super(NROWANDQN, self).__init__()

        self.env = env

        #fully connected layer with 2 hidden layers
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        # noisy layer
        self.noisy_fc3 = NoisyLinear(128, env.action_space.n)

    # forward pass
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.noisy_fc3(x)
        return x
    
    # get action
    # def act(self, state):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state).unsqueeze(0)
    #     q_value = self.forward(state)
    #     action  = q_value.max(1)[1].data[0]
    #     return action

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to("cpu")  # Ensure state is a tensor and on the correct device
        with torch.no_grad():
            q_value = self.forward(state)  # Get Q-values for all actions
        action = q_value.argmax(dim=1).item()  # Select the action with the max Q-value and convert to Python scalar
        return action
    
    def reset_noise(self):
        self.noisy_fc3.reset_noise()

    def get_sigmaloss(self):
        return self.noisy_fc3.sigma_loss()

if __name__ == '__main__':
    state_dim = 4
    action_dim = 2
    env = gym.make("CartPole-v1")
    net = NROWANDQN(state_dim, action_dim, env)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)

