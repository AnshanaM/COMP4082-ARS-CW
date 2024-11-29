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
import os

# print date and time for log file
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

        # get hyperparameters from the yaml file
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        # setting agent hyperparameters
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.initial_delta = hyperparameters['initial_delta']
        self.final_k_factor = hyperparameters['final_k_factor']
        self.noise_penalty_scale = hyperparameters['noise_penalty_scale']
        self.adam_epsilon = hyperparameters['adam_epsilon']
        self.adam_beta1 = hyperparameters['adam_beta1']
        self.adam_beta2 = hyperparameters['adam_beta2']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.max_time_steps_per_episode = hyperparameters['max_time_steps_per_episode']


        self.k_factor = 1.0

        self.loss_fn = nn.MSELoss()

        # Initialize environment
        self.env = gym.make(self.env_id)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Initialize the networks
        self.network = NROWANDQN(self.state_dim, self.action_dim, self.initial_delta).to(device)
        self.target_network = NROWANDQN(self.state_dim, self.action_dim, self.initial_delta).to(device)
        self.target_network.load_state_dict(self.network.state_dict())


        # Replay Memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Pre-fill replay memory section 5.2 page 8, "For Cartpole, MountainCar and Acrobot,..."
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        print("Pre-filling replay memory...")
        for _ in range(self.mini_batch_size):  # Pre-fill with 32 tuples
            action = self.env.action_space.sample()
            new_state, reward, terminated, _, _ = self.env.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)
            self.memory.append((state, torch.tensor(action), new_state, reward, terminated))
            state = new_state if not terminated else torch.tensor(self.env.reset()[0], dtype=torch.float, device=device)


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

    def compute_td_error(self, current_q, rewards, next_states, terminations, target_dqn):
        """
        Compute the Temporal Difference (TD) error.

        Args:
            current_q (torch.Tensor): Q-values for the current states and actions.
            rewards (torch.Tensor): Rewards for the actions taken.
            next_states (torch.Tensor): Next states resulting from the actions.
            terminations (torch.Tensor): Terminal flags for the states.
            target_dqn (nn.Module): Target Q-network.

        Returns:
            torch.Tensor: TD error.
        """
        # Compute the Q-value targets using the target network
        with torch.no_grad():
            next_q_values = target_dqn(next_states)  # Q(s_{t+1}, a; θ^−)
            max_next_q_values, _ = next_q_values.max(dim=1)  # max_a Q(s_{t+1}, a; θ^−)

            # Convert terminations to float to allow arithmetic operations
            terminations = terminations.float()

            # Target Q-value: r_t + γ * max_a Q(s_{t+1}, a; θ^−) * (1 − terminal)
            target_q = rewards + self.discount_factor_g * max_next_q_values * (1 - terminations)

        # TD error: δ_t = target_q - current_q
        td_error = target_q - current_q
        return td_error


    def compute_noise_penalty(self):
        """
        Calculate the noise penalty D for the noisy layer in the Q-network.
        
        Returns:
            noise_penalty (float): The computed noise penalty.
        """
        p_star = self.network.noisy_fc3.in_features  # Number of input features to the noisy layer
        N_a = self.network.noisy_fc3.out_features    # Number of output actions (units in the noisy layer)
        
        # Get the noise-related parameters from the noisy layer
        sigma_weights = self.network.noisy_fc3.weight_delta  # Variance of weights
        sigma_bias = self.network.noisy_fc3.bias_delta      # Variance of biases
        
        # Calculate the noise penalty (D) according to the formula in the paper
        noise_penalty = ((sigma_weights ** 2).sum() + (sigma_bias ** 2).sum()) / ((p_star + 1) * N_a)
        
        return noise_penalty

    
    def compute_total_loss(self, current_q, rewards, next_states, terminations, target_dqn):
        # Compute the TD error (standard in Q-learning)
        td_error = self.compute_td_error(current_q, rewards, next_states, terminations, target_dqn)
        
        # Compute the noise penalty D
        noise_penalty = self.compute_noise_penalty()
        
        # Combine the TD error and the noise penalty into the total loss
        td_loss = td_error.pow(2).mean()  # Standard TD error loss
        total_loss = td_loss + self.noise_penalty_scale * self.k_factor * noise_penalty
        
        return total_loss


    def sample_minibatch(self):
        """
        Sample a random minibatch from the replay memory.
        """
        minibatch = self.memory.sample(self.mini_batch_size)
        states, actions, next_states, rewards, terminations = zip(*minibatch)
        
        return torch.stack(states), torch.stack(actions), torch.stack(next_states), torch.stack(rewards), torch.tensor(terminations)


    
    def train(self):
        start_time = datetime.now()
        last_graph_update_time = start_time
        
        #print and prepare log file to write to
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')


        # run training 5 times and collect the rewards
        for run in range(1):
            # initialise epsilon, step count, rewards array for 5 training sessions, and epsilon history
            epsilon = self.epsilon_init
            step_count = 0
            rewards_per_episode_all_runs = []  # List to store rewards from all 5 runs
            epsilon_history = []

            # initialise min and max reward as positive and negative infinity
            min_reward, max_reward = float('inf'), float('-inf')

            # initialize the cumulative reward
            cumulative_reward = 0 
            # Reset rewards for each run
            rewards_per_episode = []
            print(f"\nTraining Run {run + 1}...")

            # set up environment, policy and target network
            env = self.env
            policy_dqn = self.network
            target_dqn = self.target_network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # loop for 500 episodes
            for episode in range(500):
                episode_transitions = []  # Collect all transitions for normalization
                episode_rewards = []  # Collect rewards for normalization

                for t in range(self.max_time_steps_per_episode):

                    state, _ = env.reset()
                    state = torch.tensor(state, dtype=torch.float, device=device)
                    terminated = False

                    # select action
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64, device=device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                    #execute action
                    new_state, reward, terminated, _, _ = env.step(action.item())
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                    reward = torch.tensor(reward, dtype=torch.float, device=device)

                    # store experience
                    self.memory.append((state, action, new_state, reward, terminated))


                    # calculate noise penalty D for online Q-net according to Equation 8
                    noise_penalty = self.compute_noise_penalty()

                    # log the noise penalty
                    log_message = f"Episode {episode+1}: Noise Penalty (D): {noise_penalty:.4f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')


                    # this section will update normalised rewards and calculate k according to normalised reward, max and min reward
                    # append transition, even if terminated
                    episode_rewards.append(reward)
                    episode_transitions.append((state, action, new_state, reward, terminated))
                    state = new_state
                    # normalisation
                    rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float)
                    rewards_mean = rewards_tensor.mean()
                    rewards_std = rewards_tensor.std() + 1e-6  # Avoid division by zero
                    normalized_rewards = (rewards_tensor - rewards_mean) / rewards_std
                    # Update transitions with normalized rewards
                    for idx, (state, action, new_state, _, terminated) in enumerate(episode_transitions):
                        normalized_reward = normalized_rewards[idx]
                        self.memory.append((state, action, new_state, normalized_reward, terminated))
                    # Store the cumulative reward for the episode
                    episode_reward = sum(episode_rewards)
                    rewards_per_episode.append(episode_reward)
                    # Update epsilon using decay
                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                    epsilon_history.append(epsilon)
                    # Update min/max rewards
                    min_reward = min(min_reward, episode_reward)
                    max_reward = max(max_reward, episode_reward)
                    # Update k using the equation 12
                    if max_reward - min_reward > 0:
                        k_new = self.final_k_factor * (episode_reward - min_reward) / (max_reward - min_reward)
                    else:
                        k_new = self.final_k_factor
                    # optional: smooth k to avoid drastic jumps
                    # self.k_factor = 0.9 * self.k_factor + 0.1 * k_new



                    #check if terminal state
                    if terminated:
                        # normalised reward (r+)= 0
                        normalized_reward = 0
                        state, _ = env.reset()
                        state = torch.tensor(state, dtype=torch.float, device=device)
                        terminated = False
                        pass

                    # increment step count
                    step_count += 1

                    #if step count exceeds mini batch size
                    if step_count > self.mini_batch_size:
                        # sample new random mini batch of experience from replay buffer
                        states, actions, next_states, rewards, terminations = self.sample_minibatch()
                        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze()
                        loss = self.compute_total_loss(current_q, rewards, next_states, terminations, self.target_network)
                        # updating direction of parameters is calculated
                        # update parameters according to updating direction and the learning rate
                        #optimize the policy network
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        #log the optimization step
                        log_message = f"Optimization step, Loss: {loss.item():.4f}, Step Count: {step_count}"
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')


                    
                    # sync target network with policy network parameters
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
                        

                    
                # Log episode details
                log_message = (
                    f"Episode {episode}: Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}, "
                    f"k: {self.k_factor:.4f}, Min Reward: {min_reward:.2f}, Max Reward: {max_reward:.2f}, "
                    f"Noise Penalty Scale: {self.noise_penalty_scale:.4f}, Mean Reaward: {rewards_mean}, Std Reward: {rewards_std},"
                )
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                print(log_message)
                rewards_per_episode_all_runs.append(rewards_per_episode)


        # After 5 runs, average the rewards
        mean_rewards_across_runs = np.mean(rewards_per_episode_all_runs, axis=0)
        self.save_graph(mean_rewards_across_runs, epsilon_history)
        print(f"Training completed. Saving model to {self.MODEL_FILE}")
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

        # end algorithm
                


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


    def optimize(self, policy_dqn, target_dqn):
        """
        Perform a single optimization step using the TD error and minibatch.
        """
        states, actions, next_states, rewards, terminations = self.sample_minibatch()

        # Get the current Q-values for the selected actions (from policy network)
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute the TD target (target Q-value)
        with torch.no_grad():
            target_q = rewards + self.discount_factor_g * target_dqn(next_states).max(1)[0] * (1 - terminations)

        # Compute the TD error (delta)
        td_error = target_q - current_q

        # Compute the loss (TD loss + noise penalty)
        loss = self.compute_total_loss(current_q, rewards, next_states, terminations, target_dqn)

        # Zero gradients, perform backpropagation, and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss



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
