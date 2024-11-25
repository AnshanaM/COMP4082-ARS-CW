import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory 
from dqn import DQN
from datetime import datetime, timedelta
import argparse
import itertools
import os

# print date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok = True)

#"Agg" used to generate plots as images and save to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml','r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # hyperparameters
        self.env_id = hyperparameters['env_id']
        # number of steps agent will take to replay memory
        self.network_sync_rate = hyperparameters['network_sync_rate']
        # size of replay memory
        self.replay_memory_size = hyperparameters['replay_memory_size']
        #size of the training dataset sampled from the replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size']
        # 1-100% random actions
        self.epsilon_init = hyperparameters['epsilon_init']
        # epsilon decay rate
        self.epsilon_decay = hyperparameters['epsilon_decay']
        # minimum epsilon value
        self.epsilon_min = hyperparameters['epsilon_min']
        # stop training after reaching this number of rewards
        self.stop_on_reward = hyperparameters['stop_on_reward']
        # to alter hidden dimensions for dqn
        self.fc1_nodes = hyperparameters['fc1_nodes']
        #get optional env specific parameters
        self.env_make_params = hyperparameters.get('env_make_params', {})

        # learning rate (alpha)
        self.learning_rate_a = hyperparameters['learning_rate_a']
        # discount rate (gamma)
        self.discount_factor_g = hyperparameters['discount_factor_g']

        # neural network
        # nn loss function, mse = mean squared error can be swapped
        self.loss_fn = nn.MSELoss()
        # nn optimizer
        self.optimizer = None

        # path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def train(self):
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Create an instance of the environment
        env = gym.make(self.env_id, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        memory = ReplayMemory(self.replay_memory_size)

        epsilon = self.epsilon_init
        step_count = 0
        best_reward = -9999999
        rewards_per_episode = []
        epsilon_history = []

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
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

            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0


    def test(self, render=False):
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
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

    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # save plots
        fig = plt.figure(1)

        # plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
         
        # for state, action, new_state, reward, terminted in mini_batch:

        #     if terminted:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            
        # transpose the list of experiences and separate each element

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # calculate target q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # calculate q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index = actions.unsqueeze(dim=1)).squeeze()

        # compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        #optimise the model
        self.optimizer.zero_grad()  # clear gradients
        loss.backward()             # compute gradients (backpropagation)
        self.optimizer.step()       # update network parameters i.e. weights & biases

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Testing mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.train()
    elif args.test:
        dql.test(render=True)
    else:
        print("Please specify either --train or --test.")
