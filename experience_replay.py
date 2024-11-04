# define memory for experience replay
from collections import deque
import random

class ReplayMemory():

    # maxlen to initialise the deck
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # optional seed for reproducability
        if seed is not None:
            random.seed(seed)
            
    
    # append experience to the memory
    # where the transition is (state, action, new_state, reward, terminated)
    def append(self, transition):
        self.memory.append(transition)

    # randomly samples the memory to the sample_size specified
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
