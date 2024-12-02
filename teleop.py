import torch
import gymnasium as gym
import gymnasium_robotics
import asset
import numpy as np
from HAC import HAC
import pygame

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# WASD for movement, arrow keys are reserved in mujoco and show weird behavior
KEY_ACTION_MAP = {
    pygame.K_w: np.array([0.0, 1.0]),
    pygame.K_s: np.array([0.0, -1.0]),
    pygame.K_a: np.array([-1.0, 0.0]),
    pygame.K_d: np.array([1.0, 0.0]),    
}

def teleoperate_with_keys():
    minX = float("inf")
    maxX = float("-inf")
    minY = float("inf")
    maxY = float("-inf")
    #################### Hyperparameters ####################
    max_episodes = 1000  # Number of teleop episodes
    render = True
    env_name = "PointMaze-h-v3"

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    k_level = 3
    H = 100
    threshold = np.array([0.5, 0.5, 1, 1])
    goal_state = np.array([0.48, 0.04, 0.0, 0.0])

    agent = HAC(k_level, H, state_dim, action_dim, render, threshold,
                action_bounds=None, action_offset=None,
                state_bounds=None, state_offset=None, lr=0.001)

    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # Simple display for pygame
    pygame.display.set_caption("Teleoperation: Use Arrow Keys to Move")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        goal_state = env.unwrapped.get_obs()['desired_goal']
        goal_state = np.append(goal_state, [0.0, 0.0])
        done = False

        print(f"Episode {episode}: Teleoperation in progress...")
        while not done:
            action = np.zeros(action_dim)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            for key, mapped_action in KEY_ACTION_MAP.items():
                if keys[key]:
                    action[:len(mapped_action)] += mapped_action

            next_state, reward, done, info = env.step(action)

            # Update minX, maxX, minY, maxY based on next_state
            minX = min(minX, next_state[0])
            maxX = max(maxX, next_state[0])
            minY = min(minY, next_state[1])
            maxY = max(maxY, next_state[1])

            pygame.display.flip()

            print(f"Action Taken: {action}")
            print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
            print(f"Bounds - minX: {minX}, maxX: {maxX}, minY: {minY}, maxY: {maxY}")

            agent.replay_buffer[0].add((state, action, reward, next_state, goal_state, agent.gamma, float(done)))

            state = next_state
            clock.tick(30)

        print(f"Episode {episode} Complete!")

if __name__ == "__main__":
    teleoperate_with_keys()