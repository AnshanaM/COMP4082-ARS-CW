import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import logging


def improved_td_loss(episode,frame,  batch_size, buffer, current_model, target_model, gamma, args_k, opt):
    device = "cpu"
    # sample a batch of transitions (state, action, reward, next_state, done)
    transitions = buffer.sample(batch_size)

    # unpack the batch of transitions into separate lists
    states, actions, rewards, next_states, terminated = zip(*transitions)

    # convert lists to PyTorch tensors and move to the correct device
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    terminated = torch.FloatTensor(np.array(terminated)).to(device)

    # calculate Q-values for current states
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # calculate Q-values for next states
    next_q_values = target_model(next_states)
    next_q_value = next_q_values.max(1)[0]

    # calculate expected Q-value using the Bellman equation
    expected_q_value = rewards + gamma * next_q_value * (1 - terminated)

    # calculate noise penalty (sigma loss)
    sigmaloss = current_model.get_sigmaloss()
    sigmaloss = args_k * sigmaloss

    # compute the total loss
    td_loss = (q_value - expected_q_value.detach()).pow(2).mean()
    loss = td_loss + sigmaloss

    logging.info(f"Episode: {episode}, Frame: {frame} TD Loss: {td_loss:.4f}, Sigma Loss (D): {sigmaloss:.4f}, Total Loss: {loss:.4f} k_value: {args_k:.4f}")

    # optimize the model, recalculate gradients
    opt.zero_grad()
    loss.backward()
    opt.step()

    # reset noise layers for exploration
    current_model.reset_noise()
    target_model.reset_noise()

    return loss, sigmaloss
