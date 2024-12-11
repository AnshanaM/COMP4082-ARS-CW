import gymnasium as gym
import torch
from tools import test_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu

env_id = "Pendulum-v1"

env = gym.make(env_id, render_mode="human")

# load the model and test it
model_path = 'test/pendulum.pt'

test_rewards = test_model(env, model_path, render=True, episodes=15)
print("Test Rewards:", test_rewards)