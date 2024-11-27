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
        self.env = gym.make(self.env_id)
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
            self.k_factor = max(self.k_factor * 0.99, 0.1)
            print(f"Updated k_factor: {self.k_factor:.4f}")

    def compute_total_loss(self, current_q, target_q):
        # TD-error loss (MSE)
        td_loss = self.loss_fn(current_q, target_q)

        # Compute D (noise penalty)
        p_star = self.network.noisy_fc3.in_features  # Input dimension of last layer
        N_a = self.network.noisy_fc3.out_features   # Output dimension (number of actions)
        sigma_weights = self.network.noisy_fc3.weight_sigma
        sigma_bias = self.network.noisy_fc3.bias_sigma

        # Un-normalised version
        # noise_penalty = (sigma_weights.abs().sum() + sigma_bias.abs().sum()) / ((p_star + 1) * N_a)

        # Normalize noise penalty by the dimensionality of the parameters
        noise_penalty = ((sigma_weights ** 2).sum() + (sigma_bias ** 2).sum()) / ((p_star + 1) * N_a)


        # Combine with scaling factor k
        total_loss = td_loss + self.noise_penalty_scale * self.k * noise_penalty


        # Log noise penalty
        log_message = (
            f"Noise Penalty: {noise_penalty:.4f}"
        )
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

        return total_loss

    def train(self):
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        epsilon = self.epsilon_init
        step_count = 0
        rewards_per_episode_all_runs = []  # List to store rewards from all 5 runs
        epsilon_history = []

        # Initialize reward extremes for scaling k
        min_reward, max_reward = float('inf'), float('-inf')

        # Run training 5 times and collect the rewards
        for run in range(5):
            rewards_per_episode = []  # Reset rewards for each run
            print(f"\nTraining Run {run + 1}...")

            # Initialize environment and networks for each run
            env = self.env
            policy_dqn = self.network
            target_dqn = self.target_network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            for episode in range(500):  # Train for 500 episodes
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

                    self.memory.append((state, action, new_state, reward, terminated))
                    step_count += 1
                    state = new_state

                rewards_per_episode.append(episode_reward)

                # Update epsilon using decay
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)

                # Update reward extremes
                min_reward = min(min_reward, episode_reward)
                max_reward = max(max_reward, episode_reward)

                # Update the scaling factor for noise penalty (k)
                if max_reward > min_reward:
                    self.k = self.final_k_factor * (episode_reward - min_reward) / (max_reward - min_reward + 1e-6)

                else:
                    self.k = self.final_k_factor

                # Sync target network periodically
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                # Log the episode's reward and the values of epsilon and k
                log_message = f"Episode {episode}: Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}, k: {self.k:.4f}, Noise Penalty Scale: {self.noise_penalty_scale:.4f}"
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                print(log_message)

            # After each run, store the rewards for later averaging
            rewards_per_episode_all_runs.append(rewards_per_episode)

        # After completing 5 runs, compute the average rewards for each episode
        mean_rewards_across_runs = np.mean(rewards_per_episode_all_runs, axis=0)

        # Log the average rewards across 5 runs
        with open(self.LOG_FILE, 'a') as file:
            file.write(f"\nAverage rewards after 5 runs: {mean_rewards_across_runs[:5]}...\n")

        # Save the graph with the mean rewards from all 5 runs
        self.save_graph(mean_rewards_across_runs, epsilon_history)

        print(f"Training completed. Saving model to {self.MODEL_FILE}")
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

    
    def save_graph(self, mean_rewards_across_runs, epsilon_history):
        try:
            # Print the graph file path
            print(f"Graph file path: {self.GRAPH_FILE}")

            # Ensure there's data to plot
            if len(mean_rewards_across_runs) == 0:
                print("Skipping graph save: No data available.")
                return

            # Plotting mean rewards and epsilon decay
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Create two subplots, one for mean rewards and one for epsilon decay

            # Plot the mean rewards across the 5 runs
            axs[0].plot(mean_rewards_across_runs, label="Mean Rewards (Average of 5 Runs)", color='b')
            axs[0].set_xlabel("Episodes")
            axs[0].set_ylabel("Mean Rewards")
            axs[0].set_title("Training Progress: Mean Rewards Over Episodes (Average of 5 Runs)")
            axs[0].legend()

            # Plot the epsilon decay (if available)
            if len(epsilon_history) > 0:
                axs[1].plot(epsilon_history, label="Epsilon Decay", color='r')
                axs[1].set_xlabel("Episodes")
                axs[1].set_ylabel("Epsilon Value")
                axs[1].set_title("Training Progress: Epsilon Decay")
                axs[1].legend()

            # Adjust layout to prevent overlapping
            plt.tight_layout()

            # Save the graph
            print(f"Saving graph to: {self.GRAPH_FILE}")
            fig.savefig(self.GRAPH_FILE, format="png")
            plt.close(fig)

        except Exception as e:
            print(f"Error saving graph: {e}")


    def optimize_loss(self, mini_batch, policy_dqn, target_dqn):
        # Prepare minibatch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Compute Q-values
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute total loss
        return self.compute_total_loss(current_q, target_q)

    def optimize_step(self, total_loss):
        # Perform optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def test(self, render=False):
        # Copy the environment parameters to avoid modifying the original dictionary
        env_params = self.env_make_params.copy()

        # Override render_mode if it exists in env_params
        if "render_mode" in env_params:
            env_params["render_mode"] = "human" if render else None
        else:
            env_params.update({"render_mode": "human" if render else None})

        # Create the environment with the updated parameters
        env = gym.make(self.env_id, **env_params)
        


        # Load the trained model
        policy_dqn = self.network
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()

        # Test the agent
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
