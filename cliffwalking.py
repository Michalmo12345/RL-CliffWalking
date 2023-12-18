import gymnasium as gym
import numpy as np
import random 
import seaborn as sns
import matplotlib.pyplot as plt
# The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).


it_max = 300
epsilon = 0.1
learning_rate = 0.9
discount_factor = 0.9
def q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor):
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    rewards_table = np.zeros(it_max)
    optimal_episode = None
    for episode in range(it_max):
        terminated= False
        truncated = False
        total_rewards = 0
        observation, info = env.reset()
        while not terminated and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation])
            next_observation, reward, terminated, truncated, info = env.step(action) # observation returns state, reward -1 or -100, terminated - boolean, truncated - boolean indicates time limit, info - dict unused
            total_rewards += reward
            diff = reward + discount_factor * np.max(q_table[next_observation]) - q_table[observation, action]
            q_table[observation][action] += learning_rate *diff
            observation = next_observation
        rewards_table[episode] = total_rewards
        if total_rewards == -13 and optimal_episode is None:
            optimal_episode = episode
    return q_table,rewards_table, optimal_episode


q_table, rewards_table, optimal_episode  = q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor)
print(f'Q table: {q_table}')
print(f'Rewards table: {rewards_table}')
print(f'First episode whem the path is optimal: {optimal_episode}')
print(f'Average reward: {sum(rewards_table)/it_max}')
print(f'Percentage of successes: {(np.count_nonzero(rewards_table==-13.))/np.size(rewards_table)*100}%')
# first_perfect_solution = 0
# average_reward = 0
# num_of_experiments = 10
# percentage_of_successes = 0
# for i in range(num_of_experiments):
#     q_table, rewards_table, optimal_episode  = q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor)
#     first_perfect_solution+=optimal_episode
#     average_reward += sum(rewards_table)/it_max
#     percentage_of_successes += (np.count_nonzero(rewards_table==-13.))/np.size(rewards_table)
# first_perfect_solution = first_perfect_solution/num_of_experiments
# average_reward = average_reward/num_of_experiments
# percentage_of_successes = percentage_of_successes/num_of_experiments*100
# result_first_perfect_solution = f"Average Optimal Episode: {first_perfect_solution}"
# result_average_reward = f"Average Reward: {average_reward:.2f}"
# result_percentage_of_successes = f"Percentage of Successes: {percentage_of_successes:.2f}%"
# print(result_first_perfect_solution)
# print(result_average_reward)
# print(result_percentage_of_successes)

def q_learning_optimal(learning_rate=0.1):
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    it_max = 300
    start_epsilon = 0.1
    epsilon_decay = start_epsilon/ (it_max)
    final_epsilon = 0.01
    learning_rate = learning_rate
    discount_factor = 0.9
    epsilon = start_epsilon
    epsilon_table = np.zeros(it_max)
    rewards_table = np.zeros(it_max)
    for episode in range(it_max):
        terminated= False
        truncated = False
        total_rewards = 0
        observation, info = env.reset()
        epsilon_table[episode] = epsilon
        while not terminated and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[observation])
            next_observation, reward, terminated, truncated, info = env.step(action) # observation returns state, reward -1 or -100, terminated - boolean, truncated - boolean indicates time limit, info - dict unused
            total_rewards += reward
            diff = reward + discount_factor * np.max(q_table[next_observation]) - q_table[observation, action]
            q_table[observation][action] += learning_rate *diff
            observation = next_observation
        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        rewards_table[episode] = total_rewards
    return q_table,rewards_table,epsilon_table

q_table_optimal, rewards_table_optimal, epsilon_table= q_learning_optimal()
# print(f'Q table: {q_table_optimal}')
# print(f'Rewards table: {rewards_table_optimal}')
# print(f'Average reward: {sum(rewards_table_optimal)/10000}')
# print(f'Epsilon table: {epsilon_table}')

# Test the trained agent
# env = gym.make('CliffWalking-v0', render_mode="human")

# state,info = env.reset()
# final_reward = 0
# final_path = []
# terminated = False
# truncated = False
# while not terminated and not truncated:
#     action = np.argmax(q_table[state])
#     next_state, reward, terminated,truncated,info= env.step(action)
#     final_reward += reward
#     state = next_state
#     final_path.append(state)
# # print(final_reward)
# # print(final_path)
# env.close()


#plots
def plot_learning_curve(rewards, title="Learning Curve"):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.show()

def plot_rewards_change(it_max, epsilon, initial_learning_rate, discount_factor, title="Learning rate influence"):
    num_points = 15
    learning_rates = np.linspace(initial_learning_rate, initial_learning_rate + 0.1 * (num_points - 1), num_points)
    rewards = np.zeros(num_points)
    for j in range(num_points):
        average_reward = 0
        for i in range(5):
            q_table, rewards_table, optimal_episode = q_learning_epsilon_constant(it_max, epsilon, learning_rates[j], discount_factor)
            average_reward += sum(rewards_table) / it_max
        rewards[j] = average_reward / 5

    plt.plot(learning_rates, rewards, marker='o', linestyle='-', color='r')
    plt.title(title)
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Total Rewards")
    plt.grid(True)
    plt.show()


def plot_rewards_change_optimal(title="Learning rate influence-optimal"):
    num_points = 14
    initial_learning_rate = 0.1
    learning_rates = np.linspace(initial_learning_rate, initial_learning_rate + 0.1 * (num_points - 1), num_points)
    rewards = np.zeros(num_points)
    for j in range(num_points):
        average_reward = 0
        for i in range(5):
            q_table, rewards_table, optimal_episode = q_learning_optimal(learning_rates[j])
            average_reward += sum(rewards_table) / it_max
        rewards[j] = average_reward / 5

    plt.plot(learning_rates, rewards, marker='o', linestyle='-', color='r')
    plt.title(title)
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Total Rewards")
    plt.grid(True)
    plt.show()


def plot_epsilon_change(epsilon_values, title="Epsilon change"):
    episodes = range(1, len(epsilon_values) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilon_values)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.grid(True)
    plt.show()



def plot_q_table_map(q_table):
    num_states, num_actions = q_table.shape
    q_table_reshaped = q_table.reshape((4,12, 4))  
    fig, ax = plt.subplots(figsize=(10, 4))

    for i in range(5):  
        for j in range(13):
            ax.plot([j, j], [0, 4], color='black', linewidth=0.5)  
            ax.plot([0, 12], [i, i], color='black', linewidth=0.5)  

    for i in range(4):
        for j in range(12):
            state_q_values = q_table_reshaped[i, j]
            text = '\n'.join([f'{action}: {value:.2f}' for action, value in zip(['U', 'R', 'D', 'L'], state_q_values)])
            ax.text(j + 0.5, 3.5 - i, text, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.4'),fontsize=7)

    ax.set_xticks(np.arange(0.5, 12.5, 1))
    ax.set_xticklabels(range(1, 13), fontsize=6)
    ax.set_yticks(np.arange(0.5, 4.5, 1))
    ax.set_yticklabels(range(4, 0, -1), fontsize=6)
    ax.set_xlabel('Columns', fontsize=10)
    ax.set_ylabel('Rows', fontsize=10)
    ax.set_title('Q-Values on Environment Grid', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.show()


def plot_rewards_heatmap_seaborn(it_max, epsilon, learning_rates, discount_factors, title="Learning Rate and Discount Factor Heatmap"):
    num_lr = len(learning_rates)
    num_df = len(discount_factors)

    rewards = np.zeros((num_lr, num_df))

    for i in range(num_lr):
        for j in range(num_df):
            average_reward = 0
            for k in range(5):
                q_table, rewards_table, optimal_episode = q_learning_epsilon_constant(it_max, epsilon, learning_rates[i], discount_factors[j])
                average_reward += sum(rewards_table) / it_max
            rewards[i, j] = average_reward / 5

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(rewards, annot=True, cmap="viridis", xticklabels=learning_rates, yticklabels=discount_factors, ax=ax, fmt=".2f", annot_kws={"size": 8})
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Discount Factor')
    ax.set_title(title)

    # Zmniejszenie czcionki etykiet osi X i Y
    heatmap.set_xticklabels(heatmap.get_xticklabels(), size=8)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), size=8)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Average Total Rewards', rotation=270, labelpad=15)

    plt.show()


# plot_q_table_map(q_table_optimal)
plot_epsilon_change(epsilon_table)
# plot_learning_curve(rewards_table_optimal)
# plot_rewards_change(300, 0.1, 0.1, 0.9)
# plot_rewards_change_optimal()
plot_rewards_heatmap_seaborn(300, 0.1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])