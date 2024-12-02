from scipy.stats import ttest_ind, bartlett, levene, ttest_rel
import numpy as np
import torch
import gymnasium as gym
from nrowan_dqn.original_nrowandqn import ORIGINAL_NROWANDQN
from nrowan_dqn_hybrid_noise.action_nrowandqn import ACTION_NROWANDQN

def check_and_log(p_value):
    if p_value < 0.05:
        print("There is a statistical significant difference.")
    else:
        print("There is no significant difference.")

# Perform a Student's t-test
def perform_students_ttest(data1, data2):
    """
    Perform a two-sample Student's t-test.

    Args:
    - data1 (array-like): First sample of data.
    - data2 (array-like): Second sample of data.

    Returns:
    - t_stat (float): The t-statistic.
    - p_value (float): The p-value for the test.
    """
    t_stat, p_value = ttest_ind(data1, data2, equal_var=True)
    print(f"Student's t-test: t_stat = {t_stat:.4f}, p_value = {p_value:.2e}")
    return t_stat, p_value


# Perform Bartlett's test
def perform_bartlett_test(*data_groups):
    """
    Perform Bartlett's test for homogeneity of variances.

    Args:
    - data_groups: Variable-length argument list of data arrays.

    Returns:
    - stat (float): The test statistic.
    - p_value (float): The p-value for the test.
    """
    stat, p_value = bartlett(*data_groups)
    print(f"Bartlett's test: stat = {stat:.4f}, p_value = {p_value:.2e}")
    return stat, p_value


# Perform Welch's t-test
def perform_welchs_ttest(data1, data2):
    """
    Perform Welch's t-test for unequal variances.

    Args:
    - data1 (array-like): First sample of data.
    - data2 (array-like): Second sample of data.

    Returns:
    - t_stat (float): The t-statistic.
    - p_value (float): The p-value for the test.
    """
    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    print(f"Welch's t-test: t_stat = {t_stat:.4f}, p_value = {p_value:.2e}")
    return t_stat, p_value

# Perform Levene's test
def perform_levene_test(*data_groups):
    """
    Perform Levene's test for equality of variances.

    Args:
    - data_groups: Variable-length argument list of data arrays.

    Returns:
    - stat (float): The test statistic.
    - p_value (float): The p-value for the test.
    """
    stat, p_value = levene(*data_groups, center='mean')
    print(f"Levene's test: stat = {stat:.4f}, p_value = {p_value:.2e}")
    return stat, p_value

def evaluate_model(model_path, env, model_class, *model_args):
    """
    Evaluate the model by running it in the environment for a few episodes.
    
    Args:
    - model_path (str): Path to the saved model state_dict.
    - env: Gym environment instance.
    - model_class: The class of the model to instantiate.
    - model_args: Any additional arguments required to instantiate the model.
    
    Returns:
    - rewards (list): List of total rewards for each episode.
    """
    device = "cpu"
    # Instantiate the model
    model = model_class(*model_args).to(device)
    
    # Load the saved state_dict into the model
    state_dict = torch.load(model_path, map_location=device)  # Ensure it's loaded to the correct device
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Testing {model_path}")
    rewards = []
    for ep in range(15):
        print(f"Episode: {ep + 1}")
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model.act(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    
    return rewards



# performing the tests
# Load the models
path1 = 'statistical_test_models/original.pt'
path2 = 'statistical_test_models/action_noise.pt'

# Initialize the environment
env_id = "CartPole-v1"
env = gym.make(env_id)

# Replace with your actual model class and arguments
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

rewards1 = evaluate_model(path1, env, ORIGINAL_NROWANDQN, state_dim, action_dim, env)
rewards2 = evaluate_model(path2, env, ACTION_NROWANDQN, state_dim, action_dim, env)

# Perform Statistical Tests
# Student's t-test
_, p_value = perform_students_ttest(rewards1, rewards2)
check_and_log(p_value)
# Bartlett's test
_, p_value = perform_bartlett_test(rewards1, rewards2)
check_and_log(p_value)
# Welch's t-test
_, p_value = perform_welchs_ttest(rewards1, rewards2)
check_and_log(p_value)
# Levene's test
_, p_value = perform_levene_test(rewards1, rewards2)
check_and_log(p_value)