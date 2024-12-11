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

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from diverse_priors_init import diverse_priors_init
from experience_replay import ReplayBuffer
from tools import improved_td_loss, save_graph, StatShrink2D, combined_loss_function
from action_nrowandqn import ACTION_NROWANDQN
import gymnasium as gym

# Directory setup
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUNS_DIR, 'cartpole.log')
MODEL_FILE = os.path.join(RUNS_DIR, 'cartpole.pt')
GRAPH_FILE = os.path.join(RUNS_DIR, 'cartpole.png')


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

                    loss, sigmaloss, diversity_loss_value = combined_loss_function(
                        alpha1=1,  # Set hyperparameters
                        alpha2=1,
                        beta1=0.25,
                        beta2=2,
                        states=valid_states,
                        actions=valid_actions,
                        rewards=valid_rewards,
                        next_states=valid_next_states,
                        dones=valid_dones,
                        current_model=ensemble[k],  # Current Q-function
                        target_model=target_ensemble[k],  # Corresponding target Q-function
                        gamma=discount_factor,
                        args_k=args_k,
                        opt=optimizers[k],  # Optimizer for this Q-function
                        ensemble=ensemble,  # Pass the entire ensemble
                        epsilon=0.1  # Clipping value for KL loss
                    )

                    # print(f"Updated Q-function {k + 1}: Loss={loss.item()}, Sigma Loss={sigmaloss.item()}")

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
            log_file.write(f"Episode: {episode+1},Time steps: {episode_steps},Reward: {total_reward},K final value: {args_k}\n")
            print(f"Episode: {episode+1},Time steps: {episode_steps},Reward: {total_reward},K final value: {args_k}\n")

        # Save model and graph periodically
        if episode % 10 == 0:
            torch.save(ensemble[active_model_idx].state_dict(), MODEL_FILE)
            # save_graph(episode_rewards, d_values, td_errors, k_values)
    
    # save_graph(episode_rewards, d_values, td_errors, k_values)
    print(f"Training complete. Logs saved to {LOG_FILE}, model to {MODEL_FILE}, graph to {GRAPH_FILE}.")
    return episode_rewards,d_values, td_errors, k_values

env = gym.make("CartPole-v1")
num_runs = 5
num_episodes = 500
all_rewards = []
all_d_values = []
all_td_errors = []
all_k_values = []

# Run the training 5 times
for run in range(num_runs):
    print("Run "+ str(run+1))
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Run {run}\n")
    episode_rewards, d_values, td_errors, k_values = train_bsdp(env, num_episodes, capacity=10000, batch_size=32)
    all_rewards.append(episode_rewards)
    all_d_values.append(d_values)
    all_td_errors.append(td_errors)
    all_k_values.append(k_values)


mean_td_errors, var_td_errors = StatShrink2D(all_td_errors)
mean_rewards, var_rewards = StatShrink2D(all_rewards)
mean_k_values, var_k_values = StatShrink2D(all_k_values)
mean_d_values, var_d_values = StatShrink2D(all_d_values)


save_graph(mean_rewards, var_rewards, mean_td_errors, var_td_errors, mean_k_values, var_k_values, mean_d_values, var_d_values, GRAPH_FILE)

