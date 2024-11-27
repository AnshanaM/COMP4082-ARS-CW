import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from nrowandqn import NROWANDQN
from datetime import datetime, timedelta
import argparse
import itertools
import os

# print date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

# "Agg" used to generate plots as images and save to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters from YAML
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.initial_sigma = hyperparameters['initial_sigma']
        self.final_k_factor = hyperparameters['final_k_factor']
        self.noise_penalty_scale = hyperparameters['noise_penalty_scale']
        self.adam_epsilon = hyperparameters['adam_epsilon']
        self.adam_beta1 = hyperparameters['adam_beta1']
        self.adam_beta2 = hyperparameters['adam_beta2']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


        self.loss_fn = nn.MSELoss()

        # Initialize environment
        # Do not pass render_mode during training
        self.env = gym.make(self.env_id)  # During training, no rendering
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Initialize the networks
        self.network = NROWANDQN(self.state_dim, self.action_dim, self.initial_sigma).to(device)
        self.target_network = NROWANDQN(self.state_dim, self.action_dim, self.initial_sigma).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Replay Memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate_a, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_epsilon)

        # File paths for saving models and logs
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        # Initialize optional environment parameters
        self.env_make_params = hyperparameters.get('env_make_params', {})  # Default to an empty dictionary

        # Initialize environment
        self.env = gym.make(self.env_id, **self.env_make_params)

        # Initialize NROWAN-DQN
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.network = NROWANDQN(self.state_dim, self.action_dim, sigma_init=0.017).to(device)
        self.target_network = NROWANDQN(self.state_dim, self.action_dim, sigma_init=0.017).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Initialize replay memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate_a)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.network(state).argmax().item()

    def apply_gaussian_noise(self):
        self.network.apply_gaussian_noise()

    def adjust_weights(self, total_reward, threshold=10):
        if total_reward > threshold:
            self.k_factor = max(self.k_factor * 0.99, 0.1)  # Reduce to a minimum of 0.1
            print(f"Updated k_factor: {self.k_factor:.4f}")

    def train(self):
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Create an instance of the environment (No render_mode passed here)
        env = self.env  # No render_mode here, no rendering during training

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = self.network
        target_dqn = self.target_network
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        memory = self.memory

        epsilon = self.epsilon_init
        step_count = 0
        best_reward = -9999999
        rewards_per_episode = []
        epsilon_history = []

        # Train for 500 episodes
        for episode in range(500):  # Stop after 500 episodes
            print("Episode: "+ str(episode))
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                memory.append((state, action, new_state, reward, terminated))
                step_count += 1
                state = new_state

            rewards_per_episode.append(episode_reward)

            # Log and save the model when the best reward is achieved
            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            # Update the graph every 10 seconds (or as needed)
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

            # Optimize the model if enough memory is available
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

            # Sync target network periodically
            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

        # Calculate mean rewards over the last 100 episodes
        mean_rewards = np.mean(rewards_per_episode[-100:])
        print(f"Training completed. Mean reward over the last 100 episodes: {mean_rewards}")
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)


    def save_graph(self, rewards_per_episode, epsilon_history):
        # save plots
        fig = plt.figure(1)

        # plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Compute target Q-values
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Compute current Q-values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute TD-error loss (MSE)
        td_loss = self.loss_fn(current_q, target_q)

        # Compute noise penalty (D)
        noise_penalty = 0.0
        for name, param in self.network.named_parameters():
            if "sigma" in name:  # Only include noise-related parameters
                noise_penalty += (param ** 2).sum()

        # Combine losses with noise_penalty_scale
        total_loss = td_loss + self.noise_penalty_scale * noise_penalty

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()  # Optionally return loss for logging


def test(self, render=False):
    # Set render_mode to "human" for testing
    env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
    
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_dqn = self.network
    policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
    policy_dqn.eval()

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    terminated = False
    total_reward = 0

    while not terminated:
        with torch.no_grad():
            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
        new_state, reward, terminated, _, _ = env.step(action.item())
        total_reward += reward
        state = torch.tensor(new_state, dtype=torch.float, device=device)

    print(f"Test Episode Total Reward: {total_reward}")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Testing mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)
    if args.train:
        agent.train()
    elif args.test:
        agent.test(render=True)
    else:
        print("Please specify either --train or --test.")
