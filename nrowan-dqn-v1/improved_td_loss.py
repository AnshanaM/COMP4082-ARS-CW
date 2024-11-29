import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np


def improved_td_loss(batch_size, buffer, current_model, target_model, gamma, args_k, opt):
    device = "cpu"
    # Sample a batch of transitions (state, action, reward, next_state, done)
    transitions = buffer.sample(batch_size)

    # Unpack the batch of transitions into separate lists
    states, actions, rewards, next_states, dones = zip(*transitions)

    # Convert lists to PyTorch tensors and move to the correct device
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(np.array(dones)).to(device)

    # Calculate Q-values for current states
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate Q-values for next states
    next_q_values = target_model(next_states)
    next_q_value = next_q_values.max(1)[0]

    # Calculate expected Q-value using the Bellman equation
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    # Calculate noise penalty (sigma loss)
    sigmaloss = current_model.get_sigmaloss()
    sigmaloss = args_k * sigmaloss

    # Compute the total loss
    td_loss = (q_value - expected_q_value.detach()).pow(2).mean()
    loss = td_loss + sigmaloss

    # Optimize the model
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Reset noise layers for exploration
    current_model.reset_noise()
    target_model.reset_noise()

    return loss

    device = "cpu"
    transition = buffer.sample(batch_size)

    state, action, reward, next_state, terminated = zip(*transition)

    state = torch.FloatTensor(state).to(current_model.device)
    action = torch.LongTensor(action).to(current_model.device)
    reward = torch.FloatTensor(reward).to(current_model.device)
    next_state = torch.FloatTensor(next_state).to(current_model.device)
    terminated = torch.FloatTensor(terminated).to(current_model.device)

    q_values      = current_model(state)
    next_q_values = target_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - terminated)
    
    sigmaloss = current_model.get_sigmaloss()
    # sigmaloss = sigmaloss.type(torch.cuda.FloatTensor)
    #sigmaloss   = Variable(torch.cuda.FloatTensor(np.float32(sigmaloss)).unsqueeze(1), volatile=True)
    # sigmaloss = sigmaloss.type(torch.cuda.FloatTensor)
    sigmaloss = args_k*sigmaloss

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    ##print(sigmaloss.type)
    #sigmaloss = cudaVDecoder(sigmaloss, batch_size)
    
    loss = loss + sigmaloss

    opt.zero_grad()
    loss.backward()
    opt.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss