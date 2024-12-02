# define memory for experience replay
from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    # randomly samples the memory to the sample_size specified
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
