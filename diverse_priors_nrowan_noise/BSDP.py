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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from diverse_priors_init import diverse_priors_init
from experience_replay import ReplayBuffer


def train_bsdp(env, num_episodes, capacity, batch_size):
    num_members = 5
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    learning_rate = 0.0001
    discount_factor = 0.99

    # Initialize ensemble of Q-functions
    ensemble = diverse_priors_init(num_members, input_dim, output_dim, env)
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble]

    print(f"Initializing {num_members} Q-functions with input_dim={input_dim}, output_dim={output_dim}.")

    # initialise buffer
    replay_buffer = ReplayBuffer(capacity, num_members)

    print(f"Replay buffer initialized with capacity: {capacity}.")


    # Masking distribution
    def sample_mask():
        return np.random.binomial(1, 0.5, num_members)


    # until convergence
    for _ in range(num_episodes):

        # reset env AND
        # receive state
        state = env.reset()[0]
        done = False

        print(f"Starting episode {_ + 1}/{num_episodes}. Initial state: {state}")

        # pick Q-function
        active_model_idx = np.random.randint(num_members)

        total_reward = 0

        # while episode not terminated
        while not done:
            # take action a_t = argmax Q_k (s_t, a)
            model = ensemble[active_model_idx]
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()
            

            # receive state t+1 and reward r_t from env
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward+=reward

            print(f"Action taken: {action}, Reward: {reward}, Next state: {next_state}, Done: {done}")

            # Combine termination signals
            done = terminated or truncated

            # sample bootstrap mask m_t ~ M
            mask = sample_mask()

            print(f"Sampled bootstrap mask: {mask}")


            # add (s_t, a_t, s_t+1, m_t) to replay buffer B
            replay_buffer.add((state, action, reward, next_state, done), mask)
            print(f"Added to replay buffer: State={state}, Action={action}, Next State={next_state}, Mask={mask}")


            # Update each Q-function based on its masked experience
            if len(replay_buffer.buffer) >= batch_size:
                for k in range(num_members):

                    # sample k minibatch according to masks from B to 
                    # update each member Q function by minimising TD error (2)
                    mini_batch = replay_buffer.sample(batch_size, k)
                    if not mini_batch:
                        continue
                    states, actions, rewards, next_states, dones = zip(*[
                        (exp[0], exp[1], exp[2], exp[3], exp[4]) for exp in mini_batch
                    ])
                    states = torch.FloatTensor(np.array(states))
                    actions = torch.LongTensor(np.array(actions))
                    rewards = torch.FloatTensor(np.array(rewards))
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(np.array(dones))


                    print(f"Updating Q-function {k + 1}. Mini-batch size: {len(mini_batch)}")

                    # Compute TD target
                    with torch.no_grad():
                        next_q_values = ensemble[k](next_states).max(dim=1)[0]
                        td_targets = rewards + discount_factor * next_q_values * (1 - dones)
                        print(f"TD targets: {td_targets.numpy()}")


                    # Compute TD error and update
                    q_values = ensemble[k](states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    loss = nn.MSELoss()(q_values, td_targets)
                    print(f"Loss for Q-function {k + 1}: {loss.item()}")


                    optimizers[k].zero_grad()
                    loss.backward()
                    optimizers[k].step()
                    

            state = next_state
        print(f"Episode {_ + 1} completed. Total reward: {total_reward} ##########################################################")



env = gym.make("CartPole-v1")
train_bsdp(env, 10, capacity=10000, batch_size=32)