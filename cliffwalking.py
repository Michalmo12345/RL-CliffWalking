import gymnasium as gym
import numpy as np
import random 
import seaborn as sns
import matplotlib.pyplot as plt
# The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).


it_max = 5000
epsilon = 0.1
learning_rate = 0.3
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


# q_table, rewards_table, optimal_episode  = q_learning_epsilon_constant(it_max, epsilon, learning_rate, discount_factor)
# print(f'Q table: {q_table}')
# print(f'Rewards table: {rewards_table}')
# print(f'First episode whem the path is optimal: {optimal_episode}')
# print(f'Average reward: {sum(rewards_table)/it_max}')
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

def q_learning_optimal():
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    it_max = 1000
    start_epsilon = 0.1
    epsilon_decay = start_epsilon/ (it_max)
    final_epsilon = 0.01
    learning_rate = 0.1
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
print(f'Q table: {q_table_optimal}')
print(f'Rewards table: {rewards_table_optimal}')
print(f'Average reward: {sum(rewards_table_optimal)/10000}')
print(f'Epsilon table: {epsilon_table}')

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
            ax.text(j + 0.5, 3.5 - i, text, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.15'),fontsize=7)

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

plot_q_table_map(q_table_optimal)
plot_epsilon_change(epsilon_table)
plot_learning_curve(rewards_table_optimal)
