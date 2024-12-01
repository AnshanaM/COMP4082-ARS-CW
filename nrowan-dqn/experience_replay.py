# define memory for experience replay
from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    # append experience to the memory
    # where the transition is (state, action, new_state, reward, terminated)
    # def append(self, transition):
    #     self.memory.append(transition)

    def append(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))
    
    # ALTERNATIVE TO APPEND CHECK IF MATRIX REPRESENTATION IS NEEDED
    # def push(self, state, action, reward, next_state, done):
	# 	# reshap a n-dim array as a 1*n-dim matrix
	# 	state = np.expand_dims(state, 0)
	# 	next_state = np.expand_dims(next_state, 0)
	# 	# put a 5 tuple into replay buffer
	# 	self.buffer.append((state, action, reward, next_state, done))

    # randomly samples the memory to the sample_size specified
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
