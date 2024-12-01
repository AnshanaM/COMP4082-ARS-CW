import math, random
import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn.init as init
import matplotlib.pyplot as plt
from improved_td_loss import improved_td_loss
from NoisyLinear import NoisyLinear
from nrowandqn import NROWANDQN
from experience_replay import ReplayMemory
import logging


def transpose(matrix_list):
    """
      2D list transpose, requiring the same length per row
    """
    return [[row[col] for row in matrix_list] for col in range(len(matrix_list[0]))]

def StatShrink2D(data_list):
    """
      args:
        A two-dimensional list of different lengths per line
      Function：
        based on avg_reward Calculate the mean, variance of each col
      return:
        a mean list,a var list
    """
    assert isinstance(data_list, list),"params is not list"
    assert isinstance(data_list[0], list),"params is not list2d"
    len_data = [x.__len__() for x in data_list]
    min_len = min(len_data)
    new_list = []
    for ldata in data_list:
        nlist = [ldata[index] for index in range(min_len)]
        new_list.append(nlist)
    new_list = transpose(new_list)
    mean_list = [np.mean(mdata) for mdata in new_list]
    var_list = [np.var(edata) for edata in new_list]
    return mean_list,var_list

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu

# directory for saving run info
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUNS_DIR, 'cartpole.log')
MODEL_FILE = os.path.join(RUNS_DIR, 'cartpole.pt')
GRAPH_FILE = os.path.join(RUNS_DIR, 'cartpole.png')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])


env_id = "CartPole-v1"
env = gym.make(env_id)

k_start = 0
k_final = 4
learning_rate_a = 0.0001
update_frequency = 1000
num_frames = 30000
batch_size = 32
gamma      = 0.99
N = 10000

# initialise replay memory with capacity N
replay_buffer = ReplayMemory(N)

# initialise online Q-net
current_model = NROWANDQN(env.observation_space.shape[0], env.action_space.n, env).to(device)
# initialise target Q-net
target_model  = NROWANDQN(env.observation_space.shape[0], env.action_space.n, env).to(device)

optimizer = optim.Adam(current_model.parameters(), lr=learning_rate_a)
    
update_target(current_model, target_model)

losses_all = []
rewards_all = []
k_mean = []

# observe sup(R), inf(R)
reward_inf = 0
reward_sup = 100


# k_by_reward = lambda reward_x: k_final * (reward_x - reward_inf) / (reward_sup - reward_inf)
# clamped to restrict to 0-4
k_by_reward = lambda reward_x: max(0, min(k_final, k_final * (reward_x - reward_inf) / (reward_sup - reward_inf)))
k_values_timestep = []

for i in range(1):
    num_frames = 30000
    batch_size = 32
    gamma      = 0.99

    losses = []
    all_rewards = []
    k_values = []
    d_values_sigma_losses = []
    episode_reward = 0
    current_model = NROWANDQN(env.observation_space.shape[0], env.action_space.n, env).to(device)
    target_model  = NROWANDQN(env.observation_space.shape[0], env.action_space.n, env).to(device)


    state, info = env.reset()

    optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

    replay_buffer = ReplayMemory(10000)
    update_target(current_model, target_model)

    episode = 0
    episode_reward = 0
    for frame_idx in range(1, num_frames + 1):
        args_k = 0.

        # select action
        action = current_model.act(state)

        # execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        # store experience in replay buffer
        replay_buffer.append(state, action, reward, next_state, terminated or truncated)
    
        state = next_state
    
        # accumulate reward
        episode_reward += reward

        # calculate k according to episode_reward, reward_sup and reware_inf
        args_k = k_by_reward(episode_reward)
        k_values.append(args_k)

        # logging.info(f"Frame {frame_idx}/{num_frames}, Episode {episode}, Current Reward: {reward}")

        if terminated or truncated:
            episode +=1
            # Reset environment for the next episode
            state, info = env.reset()
            all_rewards.append(episode_reward)
            logging.info(f"Episode {episode}, Frame {frame_idx}, Episode Reward: {episode_reward}, k: {args_k}")
            episode_reward = 0
            k_values_timestep.append(k_values)
            
        
        if len(replay_buffer) > batch_size:
            # calculate D  using Eq. 8 and calculate TD-error
            # sample random minibatch, implemented inside improved_td_loss function
            loss, sigma_loss = improved_td_loss(episode,frame_idx,batch_size, replay_buffer, current_model, target_model, gamma, args_k, optimizer)
            losses.append(loss.item())
            d_values_sigma_losses.append(sigma_loss.item())

            # Log loss and other information
            # logging.info(f"Episode {episode}, Frame {frame_idx}, Loss: {loss.item()}, k: {args_k}")


        # Update the target model at regular intervals
        if frame_idx % update_frequency == 0:
            update_target(current_model, target_model)
            logging.info(f"Updating target model at frame {frame_idx}, Episode {episode}, Episode Reward: {all_rewards[-1]}")
    
    losses_all.append(losses)
    rewards_all.append(all_rewards)


def save_graph(mean_rewards, mean_losses, mean_k, d_values, graph_file):
    # Create a new figure
    plt.figure(figsize=(24, 6))

    # Subplot for mean rewards
    plt.subplot(1, 4, 1)
    plt.plot(mean_rewards, label='Mean Rewards', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.title('Mean Rewards Over Episodes')
    plt.legend()

    # Subplot for mean losses
    plt.subplot(1, 4, 2)
    plt.plot(mean_losses, label='Mean Losses', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Loss')
    plt.title('Mean Losses Over Episodes')
    plt.legend()

    # Subplot for k values
    plt.subplot(1, 4, 3)
    timesteps = range(max_timesteps)
    plt.plot(timesteps, mean_k_values_timestep, label='Mean k Values', color='green')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean k Value')
    plt.title('Mean k Values Across Timesteps (Averaged over episodes)')
    plt.legend()

    # Subplot for D values (sigma losses)
    plt.subplot(1, 4, 4)
    plt.plot(d_values, label='D Values (Sigma Losses)', color='purple')
    plt.xlabel('Episodes')
    plt.ylabel('D Values')
    plt.title('D Values Over Episodes')
    plt.legend()

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(graph_file)
    plt.close()
    logging.info(f"Graph saved to {graph_file}")


# Align episodes to the longest episode length
max_timesteps = max(len(episode) for episode in k_values_timestep)
aligned_k_values = [
    episode + [None] * (max_timesteps - len(episode))  # Pad with None for short episodes
    for episode in k_values_timestep
]

# Calculate mean k values for each timestep, ignoring None
mean_k_values_timestep = []
for t in range(max_timesteps):
    timestep_values = [episode[t] for episode in aligned_k_values if episode[t] is not None]
    mean_k_values_timestep.append(np.mean(timestep_values))

logging.info(f"Training completed. Saving model to {MODEL_FILE}")
torch.save(current_model.state_dict(), MODEL_FILE)
mean_losses, var_losses = StatShrink2D(losses_all)
mean_rewards, var_rewards = StatShrink2D(rewards_all)
save_graph(mean_rewards, mean_losses, mean_k_values_timestep, d_values_sigma_losses , GRAPH_FILE)
