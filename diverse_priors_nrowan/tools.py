import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from action_nrowandqn import ACTION_NROWANDQN

def transpose(matrix_list):

    return [[row[col] for row in matrix_list] for col in range(len(matrix_list[0]))]

def StatShrink2D(data_list):

    assert isinstance(data_list, list),"params is not list"
    assert isinstance(data_list[0], list),"params is not list2d"
    len_data = [x.__len__() for x in data_list]
    min_len = min(len_data)
    new_list = []
    for ldata in data_list:
        nlist = [ldata[index] for index in range(min_len)]
        new_list.append(nlist)
    new_list = transpose(new_list)
    mean_list = [np.mean(mdata) for mdata in new_list]
    var_list = [np.var(edata) for edata in new_list]
    return mean_list,var_list

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def improved_td_loss(states, actions, rewards, next_states, dones, current_model, target_model, gamma, args_k, opt):
    device = "cpu"
    
    # Ensure all inputs are on the correct device
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Calculate Q-values for current states
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate Q-values for next states
    with torch.no_grad():
        next_q_values = target_model(next_states)
        next_q_value = next_q_values.max(1)[0]

    # Calculate expected Q-value using the Bellman equation
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    # Calculate noise penalty (sigma loss)
    sigmaloss = current_model.get_sigmaloss()
    sigmaloss = args_k * sigmaloss

    # Compute the total loss
    td_loss = (q_value - expected_q_value.detach()).pow(2).mean()
    loss = td_loss + sigmaloss

    # Optimize the model and recalculate gradients
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Reset noise layers for exploration
    current_model.reset_noise()
    target_model.reset_noise()

    return loss, sigmaloss

def save_graph(mean_rewards, var_rewards, mean_losses, var_losses, mean_k_values_timestep, var_k_values_timestep, d_values, var_d_values, graph_file):
    import math
    import matplotlib.pyplot as plt
    import logging

    # Compute standard deviations from variances
    std_rewards = [math.sqrt(v) for v in var_rewards]
    std_losses = [math.sqrt(v) for v in var_losses]
    std_k_values = [math.sqrt(v) for v in var_k_values_timestep]
    std_d_values = [math.sqrt(v) for v in var_d_values]

    # Create a new figure
    plt.figure(figsize=(20, 16))

    # Subplot for rewards
    plt.subplot(2, 2, 1)
    episodes = range(len(mean_rewards))
    plt.plot(episodes, mean_rewards, label='Mean Rewards', color='blue')
    plt.fill_between(episodes, 
                     [m - s for m, s in zip(mean_rewards, std_rewards)],
                     [m + s for m, s in zip(mean_rewards, std_rewards)],
                     color='blue', alpha=0.2, label='Variance Range')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Mean Rewards with Variance')
    plt.legend()

    # Subplot for losses
    plt.subplot(2, 2, 2)
    frames = range(len(mean_losses))
    plt.plot(frames, mean_losses, label='Mean Losses', color='red')
    plt.fill_between(frames, 
                     [m - s for m, s in zip(mean_losses, std_losses)],
                     [m + s for m, s in zip(mean_losses, std_losses)],
                     color='red', alpha=0.2, label='Variance Range')
    plt.xlabel('Frames')
    plt.ylabel('Loss')
    plt.title('Mean Losses with Variance')
    plt.legend()

    # Subplot for k values
    plt.subplot(2, 2, 3)
    timesteps = range(len(mean_k_values_timestep))
    plt.plot(timesteps, mean_k_values_timestep, label='Mean k Values', color='green')
    plt.fill_between(timesteps, 
                     [m - s for m, s in zip(mean_k_values_timestep, std_k_values)],
                     [m + s for m, s in zip(mean_k_values_timestep, std_k_values)],
                     color='green', alpha=0.2, label='Variance Range')
    plt.xlabel('Timesteps')
    plt.ylabel('k Value')
    plt.title('Mean k Values with Variance Across Timesteps')
    plt.legend()

    # Subplot for D values (sigma losses)
    plt.subplot(2, 2, 4)
    episodes = range(len(d_values))
    plt.plot(episodes, d_values, label='Mean D Values', color='purple')
    plt.fill_between(episodes, 
                     [m - s for m, s in zip(d_values, std_d_values)],
                     [m + s for m, s in zip(d_values, std_d_values)],
                     color='purple', alpha=0.2, label='Variance Range')
    plt.xlabel('Episodes')
    plt.ylabel('D Values (Sigma Loss)')
    plt.title('Mean D Values with Variance')
    plt.legend()

    # Adjust layout for more space and save the plot as a PNG file
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
    plt.savefig(graph_file)
    plt.close()
    logging.info(f"Graph saved to {graph_file}")

def test_model(env, model_path, render=True, episodes=5):
    device = "cpu"
    # Load the model architecture and weights
    model = ACTION_NROWANDQN(env.observation_space.shape[0], env.action_space.n, env).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Test the model
    rewards = []
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            # Select action using the loaded model
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()

            # Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    env.close()
    return rewards

def diversity_loss(alpha1,alpha2, model, ensemble, states, epsilon):
    kl_loss = compute_kl_loss(model, ensemble, states, epsilon)
    bd_loss = compute_bd_loss(model, states)
    nl_loss = compute_nl_loss(model, states)
    return kl_loss + alpha1 * nl_loss + alpha2 * bd_loss

def combined_loss_function(alpha1, alpha2, beta1, beta2, states, actions, rewards, next_states, dones, current_model, target_model, gamma, args_k, opt, ensemble, epsilon):
    device = "cpu"
    
    # Ensure all inputs are on the correct device
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Calculate Q-values for current states
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate Q-values for next states
    with torch.no_grad():
        next_q_values = target_model(next_states)
        next_q_value = next_q_values.max(1)[0]

    # Calculate expected Q-value using the Bellman equation
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    # TD-error loss
    td_loss = (q_value - expected_q_value.detach()).pow(2).mean()

    # Noise penalty (sigma loss)
    sigmaloss = current_model.get_sigmaloss()
    sigmaloss = args_k * sigmaloss

    # Diversity loss
    diversity_loss_value = diversity_loss(alpha1, alpha2, current_model, ensemble, states, epsilon)

    # Combine losses
    loss = beta1*(td_loss + sigmaloss) + beta2*(diversity_loss_value)

    # Optimize the model and recalculate gradients
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Reset noise layers for exploration
    current_model.reset_noise()
    target_model.reset_noise()

    return loss, sigmaloss, diversity_loss_value

def compute_kl_loss(model, ensemble, states, epsilon):
    # Get Q-values for the current model
    q_values = model(states)
    softmax_q = F.softmax(q_values, dim=-1)

    # Get median softmax Q-values for the ensemble
    ensemble_softmax_q = torch.stack([F.softmax(m(states), dim=-1) for m in ensemble], dim=0)
    median_softmax_q = torch.median(ensemble_softmax_q, dim=0).values

    # Compute the KL divergence
    kl_divergence = F.kl_div(softmax_q.log(), median_softmax_q, reduction='batchmean')

    # Clip the KL divergence
    kl_loss = -torch.clamp(kl_divergence, min=0, max=epsilon)
    return kl_loss

def compute_nl_loss(model, states, h=1e-2):
    states = states.requires_grad_(True)
    # Compute Q-values for all actions
    q_values = model(states)

    # Compute the second derivatives using finite difference
    second_derivatives = []
    for action in range(q_values.size(1)):
        grads = torch.autograd.grad(q_values[:, action].sum(), states, create_graph=True)[0]
        second_grads = torch.autograd.grad(grads.sum(), states, create_graph=True)[0]
        second_derivatives.append(second_grads)

    # Compute the modulus of the second derivatives and square them
    second_derivatives = torch.stack(second_derivatives, dim=-1)  # Shape: [batch_size, state_dim, num_actions]
    squared_second_derivatives = torch.norm(second_derivatives, dim=1) ** 2

    # Take the mean over all states and actions
    nl_loss = -squared_second_derivatives.mean()
    return nl_loss

def compute_bd_loss(model, states):
    # Compute Q-values for all actions
    q_values = model(states)

    # Compute the squared Q-values
    squared_q_values = q_values ** 2

    # Take the mean over all states and actions
    bd_loss = squared_q_values.mean()
    return bd_loss