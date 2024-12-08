# algorithm 1
'''
K = 5 (number of action candidates)
M = 5 (number of models or members in the ensemble)
can change M to 3-5 for Cartpole environment
    initialise an ensemble of priors with K members 
        AND an ensemble of trainable functions with K members
    for 0 to M:
        sample state s_i from state space S
        randomly select one model j from 1 - K
        compute Q value for sample state using selected model Q(s_i) = f_j(s_i) + p_j(s_i)
        compute median Q value of the ensemble Q(s_i) = median(f(s_i) + p(s_i))
        use gradient descent to update p_j by minimising loss function J(p_j)
    end for

loss function J(p_j) = KL_loss(epsilon) + alpha1()*NL_loss + alpha2*BD_loss

'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from action_nrowandqn import ACTION_NROWANDQN


env = gym.make('CartPole-v1')
learning_rate = 0.0001
discount_factor = 0.99
num_members = 5  # Number of NROWAN-DQN members in the ensemble

def diverse_priors_init(num_members, input_dim, output_dim, env):
    print(f"Initializing an ensemble of {num_members} Q-functions.")
    # Create ensemble
    ensemble = [ACTION_NROWANDQN(input_dim, output_dim, env) for _ in range(num_members)]
    print("Created ensemble.")
    # Initialize priors with maximum dissimilarity
    for i, model in enumerate(ensemble):
        print(f"Initializing model {i+1}.")
        for param in model.parameters():
            if param.dim() >= 2:  # Apply He initialization for weights
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            else:  # Use uniform initialization for biases or 1D tensors
                nn.init.uniform_(param, -0.1, 0.1)

    print("All models initialized.")
    return ensemble
