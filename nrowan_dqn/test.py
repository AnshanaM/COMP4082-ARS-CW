import gymnasium as gym
from tools import test_model

device = 'cpu'  # force cpu

env_id = "CartPole-v1"
# env_id = "MountainCar-v0"

env = gym.make(env_id, render_mode="human")

# load the model and test it
model_path = 'results/cartpole/latest run/cartpole.pt'
# model_path = 'runs/mountaincar.pt' # path to the saved model
test_rewards = test_model(env, model_path, render=True, episodes=15)
print("Test Rewards:", test_rewards)