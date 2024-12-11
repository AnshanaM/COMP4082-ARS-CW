import gymnasium as gym
import numpy as np
import os
import torch
import torch.optim as optim
from tools import improved_td_loss_PER, StatShrink2D, update_target, save_graph
from action_nrowandqn import ACTION_NROWANDQN
from prioritised_replay import PrioritizedReplayMemory
import logging

# Select the environment: "CartPole-v1"
env_id = "CartPole-v1"
env = gym.make(env_id)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu

# Directory for saving run info
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

# Set file names based on environment
env_name = env_id.split('-')[0].lower()
LOG_FILE = os.path.join(RUNS_DIR, f'{env_name}.log')
MODEL_FILE = os.path.join(RUNS_DIR, f'{env_name}.pt')
GRAPH_FILE = os.path.join(RUNS_DIR, f'{env_name}.png')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])

# Environment parameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

reward_inf, reward_sup = 0, 100

# Hyperparameters
k_start = 0
k_final = 4
learning_rate_a = 0.0001
update_frequency = 1000
num_frames = 30000
batch_size = 32
gamma = 0.99
N = 10000

# Initialize replay memory with capacity N
replay_buffer = PrioritizedReplayMemory(N, alpha=0.6)

# Initialize online and target Q-networks
current_model = ACTION_NROWANDQN(state_dim, action_dim, env).to(device)
target_model = ACTION_NROWANDQN(state_dim, action_dim, env).to(device)

optimizer = optim.Adam(current_model.parameters(), lr=learning_rate_a)

update_target(current_model, target_model)

losses_all = []
rewards_all = []
k_mean = []
k_values_timestep = []

# k computation based on rewards
k_by_reward = lambda reward_x: max(0, min(k_final, k_final * (reward_x - reward_inf) / (reward_sup - reward_inf)))

for i in range(5):  # Train for 5 independent runs
    losses = []
    all_rewards = []
    k_values = []
    d_values_sigma_losses = []
    episode_reward = 0

    current_model = ACTION_NROWANDQN(state_dim, action_dim, env).to(device)
    target_model = ACTION_NROWANDQN(state_dim, action_dim, env).to(device)

    state, info = env.reset()
    optimizer = optim.Adam(current_model.parameters(), lr=learning_rate_a)

    replay_buffer = PrioritizedReplayMemory(N,0.6)

    update_target(current_model, target_model)

    episode = 0
    episode_reward = 0

    for frame_idx in range(1, num_frames + 1):
        args_k = 0.0

        # Select action
        action = current_model.act(state)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Store experience in replay buffer
        replay_buffer.append(state, action, reward, next_state, terminated or truncated)

        state = next_state
        episode_reward += reward

        # Calculate k based on episode reward
        args_k = k_by_reward(episode_reward)
        k_values.append(args_k)

        if terminated or truncated:
            episode += 1
            state, info = env.reset()
            all_rewards.append(episode_reward)
            logging.info(f"Episode {episode}, Frame {frame_idx}, Reward: {episode_reward}, k: {args_k}")
            episode_reward = 0
            k_values_timestep.append(k_values)


        if len(replay_buffer) > batch_size:
            # Sample from prioritized replay buffer
            experiences, weights, indices = replay_buffer.sample(batch_size, beta=0.4)

            # Extract components
            states, actions, rewards, next_states, dones = zip(*experiences)

            # Compute TD-loss, TD-errors, and sigma loss
            td_errors, loss, sigma_loss = improved_td_loss_PER(
                episode, frame_idx, batch_size, replay_buffer, current_model, target_model,
                gamma, args_k, optimizer, weights, indices
            )

            losses.append(loss.item())
            d_values_sigma_losses.append(sigma_loss.item())

            # Update priorities in the replay buffer
            replay_buffer.update_priorities(indices, td_errors.abs().cpu().detach().numpy())


        # Update target model periodically
        if frame_idx % update_frequency == 0:
            update_target(current_model, target_model)
            logging.info(f"Updating target model at frame {frame_idx}, Episode {episode}, Reward: {all_rewards[-1]}")

    losses_all.append(losses)
    rewards_all.append(all_rewards)

# Calculate mean and variance for visualization
mean_losses, var_losses = StatShrink2D(losses_all)
mean_rewards, var_rewards = StatShrink2D(rewards_all)
mean_k_values_timestep, var_k_values_timestep = StatShrink2D(k_values_timestep)
d_values, var_d_values = StatShrink2D([d_values_sigma_losses])

# Save model and training graph
torch.save(current_model.state_dict(), MODEL_FILE)
save_graph(mean_rewards, var_rewards, mean_losses, var_losses, mean_k_values_timestep, var_k_values_timestep, d_values, var_d_values, GRAPH_FILE)
