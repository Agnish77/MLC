import numpy as np

GRID_SIZE = 10

class GridEnvironment:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.obstacles = []
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        return self.get_state()

    def get_state(self):
        return self.agent_pos

    def is_valid(self, pos):
        x, y = pos
        return (0 <= x < self.grid_size and 0 <= y < self.grid_size and pos not in self.obstacles)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            next_pos = (x, y - 1)
        elif action == 1:  # down
            next_pos = (x, y + 1)
        elif action == 2:  # left
            next_pos = (x - 1, y)
        elif action == 3:  # right
            next_pos = (x + 1, y)
        else:
            next_pos = (x, y)

        if self.is_valid(next_pos):
            self.agent_pos = next_pos

        reward = 1 if self.agent_pos == self.goal_pos else -0.1
        done = self.agent_pos == self.goal_pos

        return self.get_state(), reward, done


    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ' ')
        for ox, oy in self.obstacles:
            grid[oy][ox] = 'X'
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ay][ax] = 'A'
        grid[gy][gx] = 'G'
        print("\n".join(" ".join(row) for row in grid))
        print()
