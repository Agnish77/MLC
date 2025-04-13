import numpy as np

class QLearningAgent:
    def __init__(self, grid_size, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = np.zeros((grid_size, grid_size, num_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.num_actions = num_actions
        self.grid_size = grid_size

    def get_action(self, state):
        x, y = state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        td_target = reward + self.gamma * np.max(self.q_table[nx, ny])
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
