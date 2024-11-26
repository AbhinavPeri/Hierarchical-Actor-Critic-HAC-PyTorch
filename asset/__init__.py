from asset.continuous_mountain_car import Continuous_MountainCarEnv
from asset.pendulum import PendulumEnv
from asset.point_maze import PointMazeEnv

from gymnasium.envs.registration import register

register(
    id="MountainCarContinuous-h-v1",
    entry_point="asset:Continuous_MountainCarEnv",
)

register(
    id="Pendulum-h-v1",
    entry_point="asset:PendulumEnv",
)

register(
    id="PointMaze-h-v3",
    entry_point="asset:PointMazeEnv",
)
