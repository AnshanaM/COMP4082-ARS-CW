# algorithm 2
''' 
initialise ensemble of Q functions (Q_k) with K members using Algorithm 1
Masking distribution M, Replay Buffer B
repeat
    reset environment
    receive initial state s_0 
    pick a Q value function to follow k ~ uniform{1,...,K}
    while episode not terminated:
        take action a_t = argmax Q_k(s_t, a)
        receive state s_t+1 and reward r_t from environment
        sample bootstrap mask m_t ~ M
        add (s_t, a_t, s_t+1, m_t) to replay buffer B
        sample K mini-batch according to masks from B to update each member Q-function by minimising TD-error (2)
    end while
until convergence

m_t is a binary vector of length K that samples from a K-dimensional
Bernoulli distribution
eg: m_t = (1,0,1,1,1) means that only the second Q network cannot train on step t's data

### updating the q network minimises td error but also samples masked buffer as descibed above

'''

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import gymnasium as gym
# from diverse_priors_init import diverse_priors_init
# from experience_replay import ReplayBuffer
# from tools import improved_td_loss
# from action_nrowandqn import ACTION_NROWANDQN

# def train_bsdp(env, num_episodes, capacity, batch_size):
#     num_members = 5
#     input_dim = env.observation_space.shape[0]
#     output_dim = env.action_space.n
#     learning_rate = 0.0001
#     discount_factor = 0.99
#     target_update_freq = 500

#     # Initialize ensemble of Q-functions
#     ensemble = diverse_priors_init(num_members, input_dim, output_dim, env)
#     # target_ensemble = [torch.nn.Sequential(*list(model.children())) for model in ensemble]
#     target_ensemble = [ACTION_NROWANDQN(input_dim, output_dim, env) for model in ensemble]
#     for target_model, model in zip(target_ensemble, ensemble):
#         target_model.load_state_dict(model.state_dict())
    
#     optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble]

#     print(f"Initializing {num_members} Q-functions with input_dim={input_dim}, output_dim={output_dim}.")

#     # initialise buffer
#     replay_buffer = ReplayBuffer(capacity, num_members)

#     print(f"Replay buffer initialized with capacity: {capacity}.")

#     # Reward scaling for k-values
#     reward_inf = 0
#     reward_sup = 500  # Adjust based on the environment's reward range
#     k_by_reward = lambda reward_x: max(0, min(num_members - 1, (num_members - 1) * (reward_x - reward_inf) / (reward_sup - reward_inf)))


#     # Masking distribution
#     def sample_mask():
#         return np.random.binomial(1, 0.5, num_members)


#     step = 0
#     # until convergence
#     for _ in range(num_episodes):

#         # reset env AND
#         # receive state
#         state = env.reset()[0]
#         done = False

#         print(f"Starting episode {_ + 1}/{num_episodes}. Initial state: {state}")

#         # pick Q-function
#         active_model_idx = np.random.randint(num_members)

#         total_reward = 0

#         # while episode not terminated
#         while not done:
#             args_k = 0.

#             # take action a_t = argmax Q_k (s_t, a)
#             model = ensemble[active_model_idx]
#             with torch.no_grad():
#                 action = torch.argmax(model(torch.FloatTensor(state))).item()
            

#             # receive state t+1 and reward r_t from env
#             next_state, reward, terminated, truncated, info = env.step(action)
#             total_reward+=reward

#             # calculate k according to episode_reward, reward_sup and reware_inf
#             args_k = k_by_reward(total_reward)

#             print(f"Action taken: {action}, Reward: {reward}, Next state: {next_state}, Done: {done}")

#             # Combine termination signals
#             done = terminated or truncated

#             # sample bootstrap mask m_t ~ M
#             mask = sample_mask()

#             print(f"Sampled bootstrap mask: {mask}")


#             # add (s_t, a_t, s_t+1, m_t) to replay buffer B
#             replay_buffer.add((state, action, reward, next_state, done), mask)
#             print(f"Added to replay buffer: State={state}, Action={action}, Next State={next_state}, Mask={mask}")

#             # Algorithm implementation
#             if len(replay_buffer.buffer) >= batch_size:
#                 # Sample a single mini-batch for all Q-functions
#                 mini_batch, masks = replay_buffer.sample_batch(batch_size)

#                 # Extract data from mini-batch
#                 states, actions, rewards, next_states, dones = zip(*[
#                     (exp[0], exp[1], exp[2], exp[3], exp[4]) for exp in mini_batch
#                 ])
#                 states = torch.FloatTensor(np.array(states))
#                 actions = torch.LongTensor(np.array(actions))
#                 rewards = torch.FloatTensor(np.array(rewards))
#                 next_states = torch.FloatTensor(np.array(next_states))
#                 dones = torch.FloatTensor(np.array(dones))

#                 # Update each Q-function independently
#                 for k in range(num_members):
#                     # Filter the mini-batch based on the mask for Q-function k
#                     valid_indices = [i for i in range(batch_size) if masks[i][k] == 1]
#                     if not valid_indices:
#                         continue  # Skip if no valid experiences for this Q-function

#                     # Create a mini-batch for Q-function k
#                     valid_states = states[valid_indices]
#                     valid_actions = actions[valid_indices]
#                     valid_rewards = rewards[valid_indices]
#                     valid_next_states = next_states[valid_indices]
#                     valid_dones = dones[valid_indices]

#                     # Compute TD-error and update
#                     loss, sigmaloss = improved_td_loss(
#                         valid_states,              # Filtered states for this Q-function
#                         valid_actions,             # Filtered actions
#                         valid_rewards,             # Filtered rewards
#                         valid_next_states,         # Filtered next states
#                         valid_dones,               # Filtered done flags
#                         ensemble[k],               # Current Q-function
#                         target_ensemble[k],        # Corresponding target Q-function
#                         discount_factor,           # Gamma
#                         args_k,                    # Reward-scaling factor
#                         optimizers[k]              # Optimizer for this Q-function
#                     )

#                     print(f"Updated Q-function {k + 1}: Loss={loss.item()}, Sigma Loss={sigmaloss.item()}")

#             # Update target networks periodically
#             if step % target_update_freq == 0:
#                 for target_model, model in zip(target_ensemble, ensemble):
#                     target_model.load_state_dict(model.state_dict())


#             for model in ensemble:
#                 model.reset_noise()

#             state = next_state
#         print(f"Episode {_ + 1} completed. Total reward: {total_reward} ##########################################################")



# env = gym.make("CartPole-v1")
# train_bsdp(env, 10, capacity=10000, batch_size=32)

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from diverse_priors_init import diverse_priors_init
from experience_replay import ReplayBuffer
from tools import improved_td_loss
from action_nrowandqn import ACTION_NROWANDQN
import gymnasium as gym

# Directory setup
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUNS_DIR, 'cartpole.log')
MODEL_FILE = os.path.join(RUNS_DIR, 'cartpole.pt')
GRAPH_FILE = os.path.join(RUNS_DIR, 'cartpole.png')

def save_graph(episode_rewards, d_values, td_errors, k_values):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(d_values, label="D Values", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("D Value")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(td_errors, label="TD Errors", color="red")
    plt.xlabel("Steps")
    plt.ylabel("TD Error")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(k_values, label="K Values", color="green")
    plt.xlabel("Steps")
    plt.ylabel("K Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close()

def train_bsdp(env, num_episodes, capacity, batch_size):
    num_members = 5
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    learning_rate = 0.0001
    discount_factor = 0.99
    target_update_freq = 500

    # Initialize ensemble of Q-functions
    ensemble = diverse_priors_init(num_members, input_dim, output_dim, env)
    target_ensemble = [ACTION_NROWANDQN(input_dim, output_dim, env) for model in ensemble]
    for target_model, model in zip(target_ensemble, ensemble):
        target_model.load_state_dict(model.state_dict())
    
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble]
    replay_buffer = ReplayBuffer(capacity, num_members)

    # Metric tracking
    episode_rewards = []
    d_values = []
    td_errors = []
    k_values = []

    # Masking distribution
    def sample_mask():
        return np.random.binomial(1, 0.5, num_members)

    k_by_reward = lambda reward_x: max(0, min(num_members - 1, (num_members - 1) * reward_x / 500))

    step = 0

    # Logging
    with open(LOG_FILE, 'w') as log_file:
        log_file.write("Episode,Step,Reward,D Value,TD Error,K Value\n")

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        active_model_idx = np.random.randint(num_members)
        total_reward = 0
        episode_steps = 0

        while not done:
            # Select action
            model = ensemble[active_model_idx]
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()

            # Interact with environment
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Mask and replay buffer
            mask = sample_mask()
            replay_buffer.add((state, action, reward, next_state, done), mask)

            # Calculate k-value
            args_k = k_by_reward(total_reward)

            # Update Q-functions
            if len(replay_buffer.buffer) >= batch_size:
                mini_batch, masks = replay_buffer.sample_batch(batch_size)
                states, actions, rewards, next_states, dones = zip(*[
                    (exp[0], exp[1], exp[2], exp[3], exp[4]) for exp in mini_batch
                ])
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(np.array(dones))

                for k in range(num_members):
                    valid_indices = [i for i in range(batch_size) if masks[i][k] == 1]
                    if not valid_indices:
                        continue

                    valid_states = states[valid_indices]
                    valid_actions = actions[valid_indices]
                    valid_rewards = rewards[valid_indices]
                    valid_next_states = next_states[valid_indices]
                    valid_dones = dones[valid_indices]

                    # Compute TD-error and update
                    loss, sigmaloss = improved_td_loss(
                        valid_states,              # Filtered states for this Q-function
                        valid_actions,             # Filtered actions
                        valid_rewards,             # Filtered rewards
                        valid_next_states,         # Filtered next states
                        valid_dones,               # Filtered done flags
                        ensemble[k],               # Current Q-function
                        target_ensemble[k],        # Corresponding target Q-function
                        discount_factor,           # Gamma
                        args_k,                    # Reward-scaling factor
                        optimizers[k]              # Optimizer for this Q-function
                    )

                    print(f"Updated Q-function {k + 1}: Loss={loss.item()}, Sigma Loss={sigmaloss.item()}")

                    # Record metrics
                    td_errors.append(loss.item())
                    k_values.append(args_k)

            d_values.append(done)  # Record termination state
            state = next_state
            episode_steps += 1
            step += 1

        # Record episode metrics
        episode_rewards.append(total_reward)
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"{episode},{episode_steps},{total_reward},{done},{loss.item()},{args_k}\n")

        # Save model and graph periodically
        if episode % 10 == 0:
            torch.save(ensemble[active_model_idx].state_dict(), MODEL_FILE)
            save_graph(episode_rewards, d_values, td_errors, k_values)

    save_graph(episode_rewards, d_values, td_errors, k_values)
    print(f"Training complete. Logs saved to {LOG_FILE}, model to {MODEL_FILE}, graph to {GRAPH_FILE}.")

# Example usage
env = gym.make("CartPole-v1")
train_bsdp(env, 50, capacity=10000, batch_size=32)
