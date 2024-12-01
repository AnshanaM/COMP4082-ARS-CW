import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import math

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


def improved_td_loss(episode,frame,  batch_size, buffer, current_model, target_model, gamma, args_k, opt):
    device = "cpu"
    # sample a batch of transitions (state, action, reward, next_state, done)
    transitions = buffer.sample(batch_size)

    # unpack the batch of transitions into separate lists
    states, actions, rewards, next_states, terminated = zip(*transitions)

    # convert lists to PyTorch tensors and move to the correct device
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    terminated = torch.FloatTensor(np.array(terminated)).to(device)

    # calculate Q-values for current states
    q_values = current_model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # calculate Q-values for next states
    next_q_values = target_model(next_states)
    next_q_value = next_q_values.max(1)[0]

    # calculate expected Q-value using the Bellman equation
    expected_q_value = rewards + gamma * next_q_value * (1 - terminated)

    # calculate noise penalty (sigma loss)
    sigmaloss = current_model.get_sigmaloss()
    sigmaloss = args_k * sigmaloss

    # compute the total loss
    td_loss = (q_value - expected_q_value.detach()).pow(2).mean()
    loss = td_loss + sigmaloss

    logging.info(f"Episode: {episode}, Frame: {frame} TD Loss: {td_loss:.4f}, Sigma Loss (D): {sigmaloss:.4f}, Total Loss: {loss:.4f} k_value: {args_k:.4f}")

    # optimize the model, recalculate gradients
    opt.zero_grad()
    loss.backward()
    opt.step()

    # reset noise layers for exploration
    current_model.reset_noise()
    target_model.reset_noise()

    return loss, sigmaloss

# def save_graph(mean_rewards, mean_losses, mean_k_values_timestep, max_timesteps, d_values, graph_file):
#     # Create a new figure
#     plt.figure(figsize=(24, 6))

#     # Subplot for mean rewards
#     plt.subplot(1, 4, 1)
#     plt.plot(mean_rewards, label='Mean Rewards', color='blue')
#     plt.xlabel('Episodes')
#     plt.ylabel('Mean Rewards')
#     plt.title('Mean Rewards Over Episodes')
#     plt.legend()

#     # Subplot for mean losses
#     plt.subplot(1, 4, 2)
#     plt.plot(mean_losses, label='Mean Losses', color='red')
#     plt.xlabel('Episodes')
#     plt.ylabel('Mean Loss')
#     plt.title('Mean Losses Over Episodes')
#     plt.legend()

#     # Subplot for k values
#     plt.subplot(1, 4, 3)
#     timesteps = range(max_timesteps)
#     plt.plot(timesteps, mean_k_values_timestep, label='Mean k Values', color='green')
#     plt.xlabel('Timesteps')
#     plt.ylabel('Mean k Value')
#     plt.title('Mean k Values Across Timesteps (Averaged over episodes)')
#     plt.legend()

#     # Subplot for D values (sigma losses)
#     plt.subplot(1, 4, 4)
#     plt.plot(d_values, label='D Values (Sigma Losses)', color='purple')
#     plt.xlabel('Episodes')
#     plt.ylabel('D Values')
#     plt.title('D Values Over Episodes')
#     plt.legend()

#     # Save the plot as a PNG file
#     plt.tight_layout()
#     plt.savefig(graph_file)
#     plt.close()
#     logging.info(f"Graph saved to {graph_file}")

def save_graph(mean_rewards, var_rewards, mean_losses, var_losses, mean_k_values_timestep, var_k_values_timestep, d_values, var_d_values, graph_file):
    """
    Save graphs showing the mean and variance for rewards, losses, k values, and D values.

    Args:
    - mean_rewards, var_rewards: Mean and variance of rewards per episode.
    - mean_losses, var_losses: Mean and variance of losses per episode.
    - mean_k_values_timestep, var_k_values_timestep: Mean and variance of k values per timestep.
    - d_values, var_d_values: Mean and variance of D values per episode.
    - graph_file: Path to save the graph file.
    """
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

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(graph_file)
    plt.close()
    logging.info(f"Graph saved to {graph_file}")
