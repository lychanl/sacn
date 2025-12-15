import datetime
import json
import os

import time
from typing import Optional, List, Union, Tuple
from pathlib import Path

import gym
import gym.wrappers
import gym.wrappers.record_video
import pybullet_envs
import numpy as np
import torch as th
from torch.utils.tensorboard.writer import SummaryWriter
from gym import wrappers

from algos.base import BaseACERAgent
from algos.acer_q import ACER_Q
from algos.sac import SAC
from algos.sacn import SACN, FastSACN
from logger import CSVLogger, DefaultConsoleLogger, PeriodicConsoleLogger
from env_utils import is_atari, getPossiblyDTChangedEnvBuilder, make

ALGOS = {
    'acer_q': ACER_Q,
    'sac': SAC,
    'sacn': SACN,
    'fastsacn': FastSACN,
}


def _get_agent(algorithm: str, parameters: Optional[dict], observations_space: gym.Space,
               actions_space: gym.Space, summary_writer: SummaryWriter, device: str) -> BaseACERAgent:
    if not parameters:
        parameters = {}
    
    if algorithm not in ALGOS:
        raise NotImplemented

    return ALGOS[algorithm](observations_space=observations_space, actions_space=actions_space, summary_writer=summary_writer, device=device, **parameters)


def _get_env(env_id: str, num_parallel_envs: int, asynchronous: bool = True, seed=None) -> gym.vector.AsyncVectorEnv:
    if is_atari(env_id):
        def get_env_fn():
            return wrappers.AtariPreprocessing(
                make(env_id),
            )
        builder = get_env_fn
    else:
        builder = getPossiblyDTChangedEnvBuilder(env_id)

    def get_seeded_env(*args, **kwargs):
        env = builder(*args, **kwargs)
        if hasattr(env, 'seed'):
            env.seed(seed)
        else:
            env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    builders = [builder if seed is None else get_seeded_env for _ in range(num_parallel_envs)]
    env = gym.vector.AsyncVectorEnv(builders) if asynchronous else gym.vector.SyncVectorEnv(builders)
    return env


class Runner:

    MEASURE_TIME_TIME_STEPS = 1000

    def __init__(self, environment_name: str, algorithm: str = 'acer', algorithm_parameters: Optional[dict] = None,
                 num_parallel_envs: int = 5, evaluate_time_steps_interval: int = 1500, n_step: int = 1,
                 num_evaluation_runs: int = 5, log_dir: str = 'logs/', max_time_steps: int = -1,
                 record_end: bool = True, experiment_name: str = None, asynchronous: bool = True,
                 log_tensorboard: bool = True, do_checkpoint: bool = True, record_time_steps: int = None,
                 periodic_log: int = -1, dump=(), log_to_file_values=(), log_to_file_memory_values=(), log_to_file_act_values=(), log_to_file_steps=1000,
                 debug=False, seed=None, use_cpu=False):
        """Trains and evaluates the agent.

        Args:
            environment_name: environment to be created
            algorithm: algorithm name, one of the following: ['acer']
            algorithm_parameters: parameters of the agent
            num_parallel_envs: number of environments run in the parallel
            evaluate_time_steps_interval: number of time steps between evaluation runs, -1 if
                no evaluation should be done
            num_evaluation_runs: number of runs per one evaluation
            log_dir: logging directory
            max_time_steps: maximum number of training time steps
            record_end: True if video should be recorded after training
            asynchronous: True to use concurrent envs
            log_tensorboard: True to create TensorBoard logs
            do_checkpoint: True to save checkpoints over the training
        """
        self._elapsed_time_measure = 0
        self._time_step = 0
        self._done_episodes = 0
        self._next_evaluation_timestamp = 0
        self._next_record_timestamp = 0
        self._next_log_values_timestamp = log_to_file_steps  # don't log at the beginning
        self._n_envs = num_parallel_envs
        self._evaluate_time_steps_interval = evaluate_time_steps_interval
        self._n_step = n_step
        self._num_evaluation_runs = num_evaluation_runs
        self._max_time_steps = max_time_steps
        self._log_tensorboard = log_tensorboard
        self._do_checkpoint = do_checkpoint
        self._env_name = environment_name

        self._log_to_file_values = (log_to_file_memory_values or []) + (log_to_file_values or [])
        self._log_to_file_act_values = log_to_file_act_values
        self._log_to_file_steps = log_to_file_steps

        self._device = 'cpu' if use_cpu else 'cuda'
        self._debug = debug
        self._seed = seed

        self._record_end = record_end
        self._record_time_steps = record_time_steps
        if seed is not None:
            th.manual_seed(seed)
            np.random.seed(seed)
        self._env = _get_env(environment_name, num_parallel_envs, asynchronous, seed=seed)
        self._evaluate_env = _get_env(environment_name, num_evaluation_runs, asynchronous, seed=seed)

        self._done_steps_in_a_episode = [0] * self._n_envs
        self._returns = [0] * self._n_envs
        self._rewards = [[] for _ in range(self._n_envs)]

        experiment_name = (experiment_name + "_" if experiment_name else "") + str(os.getpid())
        self._prepare_log_dir(experiment_name, log_dir, environment_name, algorithm)

        if self._log_tensorboard:
            self.summary_writer = SummaryWriter(str(self._log_dir))
        else:
            self.summary_writer = None

        self._csv_logger = CSVLogger(
            self._log_dir / 'results.csv',
            keys=['time_step', 'eval_return_mean', 'eval_std_mean']
        )
        if self._log_to_file_values or self._log_to_file_act_values:
            if self._log_to_file_act_values is None:
                self._log_to_file_act_values = []
            if self._log_to_file_values is None:
                self._log_to_file_values = []
            self._values_csv_logger = CSVLogger(
                self._log_dir / 'values.csv', 
                keys=['time_step'] + list(self._log_to_file_values + [f'act/{v}' for v in self._log_to_file_act_values]))
        else:
            self._values_csv_logger = None
        if periodic_log > 0:
            self._logger = PeriodicConsoleLogger(periodic_log)
        else:
            self._logger = DefaultConsoleLogger()

        self._save_parameters(algorithm_parameters)
        self._agent = _get_agent(algorithm, algorithm_parameters, self._env.single_observation_space, self._env.single_action_space, self.summary_writer, self._device)
        self._current_obs, _ = self._env.reset()
        self._dump = dump

    def _prepare_log_dir(self, experiment_name, log_dir, environment_name, algorithm, num_try=0):
        if experiment_name:
            exp_name = "{experiment_name}_{num_try}" if num_try > 0 else experiment_name
            
            run_log_dir = Path(
                f"{log_dir}/{environment_name}_{algorithm}_{exp_name}"
                f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        elif num_try == 0:
            run_log_dir = Path(
                f"{log_dir}/{environment_name}_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        else:
            run_log_dir = Path(
                f"{log_dir}/{environment_name}_{algorithm}_{num_try}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        
        if run_log_dir.exists():
            self._prepare_log_dir(experiment_name, log_dir, environment_name, algorithm, num_try + 1)
        else:
            run_log_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir = run_log_dir

    def run(self):
        """Performs training. If 'evaluate' is True, evaluation of the policy is performed. The evaluation
        uses policy that is being optimized, not the one used in training (i.e. randomness is turned off)
        """
        try:
            step_logged_values = []
            logged_values = []
            while self._max_time_steps == -1 or self._time_step <= self._max_time_steps:
                if self._is_time_to_evaluate():
                    self._evaluate()
                    if self._time_step != 0:
                        self._save_results()
                        if self._do_checkpoint:
                            self._save_checkpoint()
                
                if self._is_time_to_log_values():
                    self._log_values(logged_values, step_logged_values)
                    logged_values.clear()
                    step_logged_values.clear()

                if self._is_time_to_record():
                    self.record_video()

                start_time = time.time()
                experience, step_values = self._step()
                if step_values is not None:
                    step_logged_values.append(step_values)
                self._agent.save_experience(experience)
                if self._time_step % self._n_step == 0:
                    values = self._agent.learn()
                    if values is not None:
                        logged_values.append(values)
                self._elapsed_time_measure += time.time() - start_time

                if self._time_step in self._dump:
                    self.dump()
                self._logger.timestep(self._time_step - 1)

            self._logger.flush(self._time_step - 1)
            self._csv_logger.close()
            if self._values_csv_logger:
                self._values_csv_logger.close()
            if self._record_end:
                self.record_video()
        except:
            self._logger.error()
            raise

    def _save_results(self):
        self._csv_logger.dump()
        self._logger.info(f"saved evaluation results in {self._log_dir}")

    def _step(self) -> List[Tuple[Union[int, float], np.array, float, float, bool, bool]]:
        actions, policies, log_to_file = self._agent.predict_action(self._current_obs)
        steps = self._env.step(actions)
        rewards = []
        experience = []
        old_obs = self._current_obs
        self._current_obs = steps[0]

        for i in range(self._n_envs):
            self._time_step += 1
            if self._time_step % Runner.MEASURE_TIME_TIME_STEPS == 0:
                self._measure_time()

            rewards.append(steps[1][i])
            self._done_steps_in_a_episode[i] += 1
            is_done = steps[2][i]
            is_truncated = steps[3][i]
            is_end = is_done or is_truncated

            reward = steps[1][i]
            experience.append(
                (actions[i], old_obs[i], self._current_obs[i], reward, policies[i], is_done, is_end)
            )

            self._returns[i] += steps[1][i]
            self._rewards[i].append(steps[1][i])

            if is_end:
                self._done_episodes += 1
                self._logger.episode_finish(self._time_step, self._done_episodes, self._returns[i])
                if self.summary_writer:
                    self.summary_writer.add_scalar('rewards/return', self._returns[i], self._time_step)
                    self.summary_writer.add_scalar('rewards/episode length', self._done_steps_in_a_episode[i], self._time_step)

                self._returns[i] = 0
                self._rewards[i] = []
                self._done_steps_in_a_episode[i] = 0

        self._current_obs = np.array(self._current_obs)
        return experience, log_to_file

    def _evaluate(self):
        self._next_evaluation_timestamp += self._evaluate_time_steps_interval

        returns = [0] * self._num_evaluation_runs
        envs_finished = [False] * self._num_evaluation_runs
        time_step = 0
        current_obs, _ = self._evaluate_env.reset()

        while not all(envs_finished):
            time_step += 1
            actions, _, _ = self._agent.predict_action(current_obs, is_deterministic=True)
            steps = self._evaluate_env.step(actions)
            current_obs = steps[0]
            for i in range(self._num_evaluation_runs):
                if not envs_finished[i]:
                    returns[i] += steps[1][i]

                    is_done = steps[2][i]
                    is_truncated = steps[3][i]

                    envs_finished[i] = is_done or is_truncated

        self._logger.evaluation_results(self._time_step, returns)
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        if self.summary_writer:
            self.summary_writer.add_scalar('rewards/evaluation_return_mean', mean_returns, self._time_step)
            self.summary_writer.add_scalar('rewards/evaluation_return_std', std_returns, self._time_step)

        self._csv_logger.log_values(
            {'time_step': self._time_step, 'eval_return_mean': mean_returns, 'eval_std_mean': std_returns}
        )

    def record_video(self):
        if self._record_time_steps:
            self._next_record_timestamp += self._record_time_steps
        self._logger.info(f"saving video...")
        try:
            env = gym.wrappers.record_video.RecordVideo(make(self._env_name), self._log_dir / f'video-{self._time_step}')
            is_end = False
            time_step = 0
            current_obs, _ = np.array([env.reset()])

            while not is_end:
                time_step += 1
                actions, _, _ = self._agent.predict_action(current_obs, is_deterministic=True)
                obs, reward, done, truncated, _ = env.step(actions[0])
                current_obs = np.array([obs])

                is_end = done or truncated

            env.close()
            self._logger.info(f"saved video in {str(self._log_dir / f'video-{self._time_step}')}")
        except Exception as e:
            self._logger.info(f"Error while recording the video. Make sure you've got proper drivers"
                          f"and libraries installed (i.e ffmpeg). Error message:\n {e}")

    def _is_time_to_evaluate(self):
        return self._evaluate_time_steps_interval != -1 and self._time_step >= self._next_evaluation_timestamp

    def _is_time_to_record(self):
        return self._record_time_steps is not None and self._time_step >= self._next_record_timestamp

    def _measure_time(self):
        if self.summary_writer:
            self.summary_writer.add_scalar(
                'acer/time steps per second',
                Runner.MEASURE_TIME_TIME_STEPS / self._elapsed_time_measure,
                self._time_step
            )
        self._elapsed_time_measure = 0

    def _is_time_to_log_values(self):
        return (
            self._log_to_file_values or self._log_to_file_act_values
        ) and self._time_step >= self._next_log_values_timestamp

    def _log_values(self, values, step_values):
        self._next_log_values_timestamp += self._log_to_file_steps

        if not values and not step_values:
            return

        means_dict = {}
        if values:
            means = np.nanmean(values, 0)
            means_dict.update({name: value for name, value in zip(self._log_to_file_values, means)})
        if step_values:
            step_means = np.mean(step_values, 0)
            means_dict.update(
                {f'act/{name}': value for name, value in zip(self._log_to_file_act_values, step_means)}
            )
        means_dict['time_step'] = self._time_step

        self._values_csv_logger.log_values(means_dict)
        self._values_csv_logger.dump()
        self._logger.log_values(means_dict)

    def _save_parameters(self, algorithm_parameters: dict):
        with open(str(self._log_dir / 'parameters.json'), 'wt') as f:
            json.dump(algorithm_parameters, f)

    def _save_checkpoint(self):
        """Saves current state and model"""
        checkpoint_dir = self._log_dir / 'checkpoint'
        checkpoint_dir.mkdir(exist_ok=True)

        runner_dump = {
            'time_step': self._time_step,
            'done_episodes': self._done_episodes,
        }
        with open(str(checkpoint_dir / 'runner.json'), 'wt') as f:
            json.dump(runner_dump, f)
        os.makedirs(checkpoint_dir / 'model', exist_ok=True)
        self._agent.save(checkpoint_dir / 'model')

        self._logger.info(f"saved checkpoint in '{str(checkpoint_dir)}'")

    def dump(self):
        dump_dir = self._log_dir / f'dump_{self._time_step}'
        dump_dir.mkdir()

        self._agent.save(dump_dir / 'model')
