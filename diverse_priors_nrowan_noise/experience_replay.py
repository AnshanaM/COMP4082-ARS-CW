import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from diverse_priors_init import diverse_priors_init

class ReplayBuffer:
    def __init__(self, capacity, num_members):
        self.capacity = capacity
        self.buffer = []
        self.num_members = num_members

    def add(self, experience, mask):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append((experience, mask))

    def sample(self, batch_size, member_index):
        # Filter experiences based on the mask
        valid_experiences = [exp for exp, mask in self.buffer if mask[member_index] == 1]

        # Ensure there are enough valid experiences
        if len(valid_experiences) < batch_size:
            raise ValueError(f"Not enough valid experiences for member {member_index}. Available: {len(valid_experiences)}")

        # Randomly sample experiences
        sampled_indices = np.random.choice(len(valid_experiences), batch_size, replace=False)
        sampled_experiences = [valid_experiences[i] for i in sampled_indices]

        # Extract and convert components to numpy arrays
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )
