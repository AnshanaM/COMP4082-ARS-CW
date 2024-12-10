import gymnasium as gym
from tools import test_model

device = 'cpu'  # force cpu

env_id = "Pendulum-v1"
# env_id = "MountainCar-v0"

env = gym.make(env_id, render_mode="human")

# load the model and test it
# model_path = 'results/cartpole/latest run/cartpole.pt'
model_path = 'results/pendulum/60000frames/pendulum.pt' # path to the saved model
test_rewards = test_model(env_id, model_path, render=True, episodes=15)
print("Test Rewards:", test_rewards)