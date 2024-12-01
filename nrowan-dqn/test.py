import gymnasium as gym
import torch
from tools import test_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu

env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="human")

# load the model and test it
model_path = 'results/5 runs with mean and variance/cartpole.pt' # path to the saved model
test_rewards = test_model(env, model_path, render=True, episodes=5)
print("Test Rewards:", test_rewards)