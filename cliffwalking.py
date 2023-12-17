import gymnasium as gym
import numpy as np
import random 
import seaborn as sns
import matplotlib.pyplot as plt
# The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).



def q_learning_epsilon_constant():
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    it_max = 10000
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.9
    rewards_table = np.zeros(it_max)
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
    return q_table,rewards_table
q_table, rewards_table = q_learning_epsilon_constant()
print(q_table)
print(rewards_table)



def q_learning_optimal():
    env = gym.make('CliffWalking-v0', render_mode="rgb_array")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    it_max = 10000
    start_epsilon = 0.5
    epsilon_decay = start_epsilon/ (it_max / 2)
    final_epsilon = 0.05
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
    return q_table,rewards_table

# Test the trained agent
env = gym.make('CliffWalking-v0', render_mode="human")

state,info = env.reset()
final_reward = 0
final_path = []
terminated = False
truncated = False
while not terminated and not truncated:
    action = np.argmax(q_table[state])
    next_state, reward, terminated,truncated,info= env.step(action)
    final_reward += reward
    state = next_state
    final_path.append(state)
print(final_reward)
print(final_path)
env.close()


