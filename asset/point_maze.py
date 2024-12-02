from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
from gymnasium.core import RenderFrame


class PointMazeEnv(gym.Env):
    def __init__(self):
        super(PointMazeEnv, self).__init__()
        self.env = gym.make("PointMaze_UMaze-v3", render_mode="human")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space['observation']
        self.cur_obs = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.cur_obs = observation
        # Extract the necessary information from the observation dictionary
        # Assuming the observation dictionary has a key 'observation' that contains the numpy array
        observation = observation['observation']
        return observation, reward, terminated or truncated, info

    def get_obs(self):
        return self.cur_obs

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.cur_obs = observation
        # Extract the necessary information from the observation dictionary
        observation = observation['observation']
        return observation