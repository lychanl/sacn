from functools import partial
import gym.version
import gym.wrappers
import gym.wrappers.compatibility
import gym.wrappers.env_checker
import numpy as np
import importlib
import re

import gym
from collections import UserDict

import pybullet_envs  # make sure bullet envs are registered
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene


def make(env_id: str, *args, **kwargs) -> bool:
    if 'BulletEnv' in env_id and int(gym.version.VERSION.split('.')[1]) > 20:
        env = gym.make(env_id, apply_api_compatibility=True, *args, **kwargs)
        print(env)
        return env
    else:
        return gym.make(env_id, *args, **kwargs)


def is_atari(env_id: str) -> bool:
    """Checks if environments if of Atari type
    Args:
        env_id: name of the environment
    Returns:
        True if its is Atari env
    """
    env_specs = [spec for env, spec in gym.envs.registration.registry.items() if env == env_id]
    if not env_specs:
        return False
    env_spec = env_specs[0]

    if not isinstance(env_spec.entry_point, str):
        return False
    env_type = env_spec.entry_point.split(':')[0].split('.')[-1]
    return env_type == 'atari'


def get_env_variables(env):
    """Returns OpenAI Gym environment specific variables like action space dimension"""
    if type(env.observation_space) == gym.spaces.discrete.Discrete:
        observations_dim = env.observation_space.n
    else:
        observations_dim = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        continuous = False
        actions_dim = env.action_space.n
        action_scale = 1
    else:
        continuous = True
        actions_dim = env.action_space.shape[0]
        action_scale = np.maximum(env.action_space.high, np.abs(env.action_space.low))
    max_steps_in_episode = env.spec.max_episode_steps
    return action_scale, actions_dim, observations_dim, continuous, max_steps_in_episode


def getDTChangedEnvName(base_env_name, timesteps_increase, keep_ts_reward=False):
    rew_part = 'KR' if keep_ts_reward else ''
    return str.join('TS' + str(timesteps_increase) + rew_part + '-', base_env_name.split('-'))


def getPossiblyDTChangedEnvBuilder(env_name):
    prog = re.compile(r'(\w+)TS(\d+)(KR)?\-v(\d+)')
    match = prog.fullmatch(env_name)
    if not match:
        return partial(make, env_name)

    name = match.group(1)
    timesteps_increase = int(match.group(2))
    keep_reward = match.group(3)
    version = match.group(4)

    base_name = f'{name}-v{version}'
    
    base_spec = gym.envs.registration.registry.env_specs[base_name]

    mod_name, attr_name = base_spec.entry_point.split(":")
    mod = importlib.import_module(mod_name)
    base_class = getattr(mod, attr_name)

    class TimeStepChangedEnv(base_class):
        def create_single_player_scene(self, bullet_client):
            self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                        gravity=9.8,
                                                        timestep=0.0165 / timesteps_increase / 4,
                                                        frame_skip=4)
            return self.stadium_scene
    
    def builder():
        env = gym.wrappers.compatibility.EnvCompatibility(TimeStepChangedEnv())

        if not keep_reward:
            env = gym.wrappers.TransformReward(env, lambda r: r / timesteps_increase)

        if base_spec.max_episode_steps:
            env = gym.wrappers.TimeLimit(env, base_spec.max_episode_steps * timesteps_increase)
        
        return env
    
    return builder


