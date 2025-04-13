from environment import GridEnvironment, GRID_SIZE
from agent import QLearningAgent
import time

NUM_ACTIONS = 4
NUM_EPISODES = 2000
MAX_STEPS = 100

def fixed_obstacles():
    return [
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (3, 5), (4, 5), (5, 5), (6, 5), (7, 5),
        (7, 6), (7, 7), (7, 8)
    ]

def train_agent():
    env = GridEnvironment(GRID_SIZE)
    env.obstacles = fixed_obstacles()
    agent = QLearningAgent(GRID_SIZE, NUM_ACTIONS)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        for step in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        if episode % 50 == 0:
            print(f"Episode {episode} completed in {step + 1} steps")
    return agent

def visualize_agent(agent):
    env = GridEnvironment(GRID_SIZE)
    env.obstacles = fixed_obstacles()
    state = env.reset()
    agent.epsilon = 0  # Use best learned policy

    print("🔁 Agent navigating...\n")
    for step in range(200):
        env.render()
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        time.sleep(0.3)
        state = next_state
        if done:
            print(f"\n🎉 Agent successfully reached the goal in {step + 1} steps!")
            env.render()
            return
    print("\n❌ Agent failed to reach the goal.")

if __name__ == "__main__":
    agent = train_agent()
    visualize_agent(agent)
