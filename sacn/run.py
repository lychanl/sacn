import argparse
import signal

from collections import UserDict
from sys import argv
import gym

registry = UserDict(gym.envs.registration.registry)
registry.env_specs = registry
gym.envs.registration.registry = registry
gym.envs.registry = registry

import torch as th

if '--debug' in argv or th.__version__.startswith('1.'):
    th.compile = lambda x=None, *args, **kwargs: x if x else th.compile

from numpy import float32

import re
from runners import Runner, ALGOS
from env_utils import getDTChangedEnvName, make


def prepare_parser():
    parser = argparse.ArgumentParser(description='BaseActor-Critic with experience replay.', allow_abbrev=False)
    parser.add_argument('--defaults', type=str, default=None)
    parser.add_argument('--env', type=str, help='OpenAI Gym environment name', default="HalfCheetahBulletEnv-v0")
    args, _ = parser.parse_known_args()

    defaults = []
    if args.defaults:
        with open(args.defaults) as defaults_file:
            defaults_str = defaults_file.read()
            defaults_lines = []
            for line in defaults_str.splitlines():
                if line.startswith('('):
                    env_spec = line[1:line.rfind(')')]
                    if re.fullmatch(env_spec, args.env):
                        defaults_lines.append(line[line.rfind(')') + 1:])
                else:
                    defaults_lines.append(line)
            defaults = ' '.join(defaults_lines).split()

    parser.add_argument('--algo', type=str, help='Algorithm to be used', default="fastacer", choices=ALGOS)

    parser.add_argument('--evaluate_time_steps_interval', type=int, help='Number of time steps between evaluations. '
                                                                        '-1 to turn evaluation off',
                        default=10000)
    parser.add_argument('--num_evaluation_runs', type=int, help='Number of evaluation runs in a single evaluation',
                        default=10)
    parser.add_argument('--max_time_steps', type=int, help='Maximum number of time steps of agent learning. -1 means no '
                                                        'time steps limit',
                        default=-1)
    parser.add_argument('--log_dir', type=str, help='Logging directory', default='logs/')
    parser.add_argument('--nan_guard', help='Debug help', action='store_true')
    parser.add_argument('--no_checkpoint', help='Disable checkpoint saving', action='store_true')
    parser.add_argument('--no_tensorboard', help='Disable tensorboard logs', action='store_true')
    parser.add_argument('--log_values', help='Log values during training', type=str, nargs='*')
    parser.add_argument('--log_act_values', help='Log values during action prediction', type=str, nargs='*')
    parser.add_argument('--log_memory_values', help='Log values during training during memory update step (if any)', type=str, nargs='*')
    parser.add_argument('--log_to_file_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_memory_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_act_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_steps', help='Log values during training to a file every n steps, averaged over these steps', type=int, default=1000)
    parser.add_argument('--experiment_name', type=str, help='Name of the current experiment', default='')
    parser.add_argument('--save_video_on_kill', action='store_true',
                        help='True if SIGINT signal should trigger registration of the video')
    parser.add_argument('--record_time_steps', type=int, default=None,
                        help='Number of time steps between evaluation video recordings')
    parser.add_argument('--use_cpu', action='store_true',
                        help='True if CPU (instead of GPU) should be used')
    parser.add_argument('--synchronous', action='store_true',
                        help='True if not use asynchronous envs')
    parser.add_argument('--timesteps_increase', help='Timesteps per second increase. Affects gamma, max time steps and buffer size', type=int, default=None)
    parser.add_argument('--keep', help='List of parameters to do not autoadapt if timesteps_increase > 1', type=str, nargs='*', default=[])

    parser.add_argument('--dump', help='Dump memory and models on given timesteps', nargs='*', type=int)
    parser.add_argument('--debug', help='Disable tf functions', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--force_periodic_log', help='Force logging every n timesteps instead of on episode finish etc', type=int, default=0)
    parser.add_argument('--n_step', type=int, help='experience replay frequency', required=False, default=1)
    parser.add_argument('--num_parallel_envs', type=int, help='Number of environments to be run in a parallel', default=1)
    parser.add_argument('--num_threads', type=int, help='Number of threads for inter- and intra-op parallelism', default=None)
    parser.add_argument('--keep_ts_reward', default=False, action='store_true', help='If timesteps_increase, don\'t scale reward')
    default_args_partial, _ = parser.parse_known_args(defaults)
    parser.set_defaults(**default_args_partial.__dict__)
    partial_args, unparsed = parser.parse_known_args()

    sample_env = make(partial_args.env)
    ALGOS[partial_args.algo].prepare_parser(parser, sample_env, partial_args, defaults, unparsed)
    try:
        default_args = parser.parse_args(defaults)
    except Exception as e:
        raise Exception("An exception occured while parsing defaults file") from e

    parser.set_defaults(**default_args.__dict__)
    return parser


runner = None


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    parameters = {k: v for k, v in args.__dict__.items() if v is not None}
    parameters.pop('env')
    evaluate_time_steps_interval = parameters.pop('evaluate_time_steps_interval')
    n_step = parameters.pop('n_step')
    num_evaluation_runs = parameters.pop('num_evaluation_runs')
    max_time_steps = parameters.pop('max_time_steps')
    save_video_on_kill = parameters.pop('save_video_on_kill')
    no_checkpoint = parameters.pop('no_checkpoint')
    no_tensorboard = parameters.pop('no_tensorboard')
    record_time_steps = parameters.pop('record_time_steps', None)
    experiment_name = parameters.pop('experiment_name')
    algorithm = parameters.pop('algo')
    log_dir = parameters.pop('log_dir')
    use_cpu = parameters.pop('use_cpu')
    synchronous = parameters.pop('synchronous')
    env_name = args.env
    log_to_file_values = args.log_to_file_values
    log_to_file_memory_values = args.log_to_file_memory_values
    log_to_file_act_values = args.log_to_file_act_values
    log_to_file_steps = parameters.pop('log_to_file_steps')
    debug = parameters.pop('debug')
    dump = args.dump or ()
    seed = parameters.pop('seed', None)
    num_threads = parameters.pop('num_threads', None)

    if num_threads is not None:
        th.set_num_threads(num_threads)
        th.set_num_interop_threads(num_threads)

    timesteps_increase = parameters.pop('timesteps_increase', None)
    if timesteps_increase and timesteps_increase != 1:
        keep_ts_reward = parameters.pop('keep_ts_reward')
        env_name = getDTChangedEnvName(env_name, timesteps_increase, keep_ts_reward)
        if 'max_time_steps' not in args.keep:
            max_time_steps *= timesteps_increase
            print(f'Auto-adapted max_time_steps to {max_time_steps}')
        if 'evaluate_time_steps_interval' not in args.keep:
            evaluate_time_steps_interval *= timesteps_increase
            print(f'Auto-adapted evaluate_time_steps_interval to {evaluate_time_steps_interval}')
        if 'n_step' not in args.keep:
            n_step = n_step * timesteps_increase
            print(f'Auto-adapted n_step to {n_step}')

    if experiment_name:
        for param, value in parameters.items():
            experiment_name = experiment_name.replace(f'{{{param}}}', str(value))

    runner = Runner(
        environment_name=env_name,
        algorithm=algorithm,
        algorithm_parameters=parameters,
        num_parallel_envs=args.num_parallel_envs,
        log_dir=log_dir,
        max_time_steps=max_time_steps,
        num_evaluation_runs=num_evaluation_runs,
        evaluate_time_steps_interval=evaluate_time_steps_interval,
        experiment_name=experiment_name,
        asynchronous=not synchronous,
        log_tensorboard=not no_tensorboard,
        do_checkpoint=not no_checkpoint,
        record_time_steps=record_time_steps,
        n_step=n_step,
        dump=dump,
        periodic_log=args.force_periodic_log,
        log_to_file_values=log_to_file_values,
        log_to_file_act_values=log_to_file_act_values,
        log_to_file_steps=log_to_file_steps,
        log_to_file_memory_values=log_to_file_memory_values,
        debug=debug,
        seed=seed,
        use_cpu=use_cpu
    )

    import run
    run.runner = runner

    def handle_sigint(sig, frame):
        runner.record_video()

    if save_video_on_kill:
        signal.signal(signal.SIGINT, handle_sigint)

    runner.run()


if __name__ == "__main__":
    main()
