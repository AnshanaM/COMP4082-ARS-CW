# # define memory for experience replay
# from collections import deque
# import random

# class ReplayMemory():
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)

#     def append(self, state, action, reward, next_state, terminated):
#         self.memory.append((state, action, reward, next_state, terminated))

#     # randomly samples the memory to the sample_size specified
#     def sample(self, sample_size):
#         return random.sample(self.memory, sample_size)

#     def __len__(self):
#         return len(self.memory)


import numpy as np
from collections import deque
import random

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize memory for prioritized experience replay.
        Args:
        - capacity (int): Maximum size of the replay memory.
        - alpha (float): Determines how much prioritization is used.
          Alpha = 0 corresponds to uniform sampling (no prioritization).
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Store priorities for each experience
        self.alpha = alpha

    def append(self, state, action, reward, next_state, terminated):
        """
        Add an experience to memory.
        Assign the maximum priority to newly added experiences.
        """
        max_priority = max(self.priorities, default=1.0)  # Default priority if memory is empty
        self.memory.append((state, action, reward, next_state, terminated))
        self.priorities.append(max_priority)

    def sample(self, sample_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.
        Args:
        - sample_size (int): Number of experiences to sample.
        - beta (float): Importance-sampling exponent to correct for bias.

        Returns:
        - Sampled batch of experiences.
        - Importance-sampling weights for each sampled experience.
        - Indices of the sampled experiences.
        """
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty memory!")

        # Convert priorities to probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probs = scaled_priorities / sum(scaled_priorities)

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), sample_size, p=sampling_probs)

        # Retrieve sampled experiences
        sampled_experiences = [self.memory[idx] for idx in indices]

        # Compute importance-sampling weights
        total_experiences = len(self.memory)
        weights = (total_experiences * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return sampled_experiences, weights, indices

    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences based on TD-errors.
        Args:
        - indices (list of int): Indices of the experiences to update.
        - td_errors (list of float): Corresponding TD-errors for each experience.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Add a small constant to avoid zero priority

    def __len__(self):
        return len(self.memory)

