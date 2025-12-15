from typing import Optional, List, Union, Dict, Tuple
import functools
import gym
import torch as th
import numpy as np

from algos.base import BaseACERAgent, BaseActor, Critic
from replay_buffer import BufferFieldSpec


def print_batch(batch):
    if batch is None:
        return

    def print_el(el):
        if len(np.shape(el)) == 0:
            if isinstance(el, float) or isinstance(el, np.ndarray) and el.dtype == np.float32:
                return f'{el:.2e}'
            return str(el)
        else:
            return f"[{','.join(map(print_el, el)) if np.shape(el)[0] < 10 else ','.join(map(print_el, el[:10])) + '...'}]"

    keys = []
    vals = []
    for k, v in batch.items():
        keys.append(k)
        vals.append(v.detach().cpu().numpy() if isinstance(v, th.Tensor) else v)
    
    print(*keys, sep='\t')

    for row in zip(*vals):
        print(*map(print_el, row), sep='\t')


class BaseNextGenACERAgent(BaseACERAgent):
    @classmethod
    def prepare_parser(cls, parser, sample_env, base_args, default_args, unparsed_args):
        preproc_params = cls.get_args_preproc_params(base_args, default_args, unparsed_args)
        args = cls.get_args(preproc_params, [])
        args['actor_type'] = (str, 'simple', {'choices': cls.ACTORS.keys()})
        args['critic_type'] = (str, 'simple', {'choices': cls.CRITICS.keys()})
        args['buffer_type'] = (str, 'simple', {'choices': cls.BUFFERS.keys()})

        discrete_env = isinstance(sample_env.action_space, gym.spaces.Discrete)
        args.update(cls.get_component_args('actor', discrete_env, cls.ACTORS, default_args, unparsed_args, preproc_params))
        args.update(cls.get_component_args('critic', None, cls.CRITICS, default_args, unparsed_args, preproc_params))
        args.update(cls.get_component_args('buffer', 0, cls.BUFFERS, default_args, unparsed_args, preproc_params))

        for name, arg in args.items():
            if len(arg) == 2:
                type, default = arg
                kwargs = {}
            else:
                type, default, kwargs = arg
            flag = kwargs.get('action', None) == 'store_true'
            if flag:
                parser.add_argument(f'--{name}', default=default, **kwargs)
            else:
                parser.add_argument(f'--{name}', type=type, default=default, **kwargs)

    @classmethod
    def get_args_preproc_params(cls, base_args, defaults, args):
        params = {}
        if 'timesteps_increase' in base_args and base_args.timesteps_increase is not None:
            params['timesteps_increase'] = base_args.timesteps_increase
        if 'keep' in base_args:
            params['keep'] = base_args.keep
        return params

    @classmethod
    def get_component_class(cls, component, component_extra_type_param, available, defaults, args):
        component_type = 'simple'
        if f'--{component}_type' in defaults:
            component_type = defaults[defaults.index(f'--{component}_type') + 1]
        if f'--{component}_type' in args:
            component_type = args[args.index(f'--{component}_type') + 1]

        if component_type not in available:
            raise KeyError(
                f'{component_type} is not a valid {component} for this algo. Choices: '
                + ', '.join(available.keys())
            )
        component_cls = available[component_type]
        if component_extra_type_param is not None:
            component_cls = component_cls[component_extra_type_param]
        return component_cls

    @classmethod
    def get_component_args(cls, component, component_extra_type_param, available, defaults, args, preproc_params):
        component_cls = cls.get_component_class(component, component_extra_type_param, available, defaults, args)
        component_args = component_cls.get_args(preproc_params, [component])
        return {f'{component}.{name}': arg for name, arg in component_args.items()}

    ACTORS = {}
    BUFFERS = {}
    CRITICS = {}

    DATA_FIELDS = ('lengths', 'obs', 'obs_next', 'actions', 'policies', 'rewards', 'dones', 'priorities', 'n', 'batch_size', 'time')
    ACT_DATA_FIELDS = ('obs', 'actions')
    CONTROL_FIELDS = ('n', 'batch_size')

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space,
                 update_blocks=1, buffer_type='simple', log_values=(), log_memory_values=(), log_act_values=(),
                 log_to_file_values=(), log_to_file_memory_values=(), log_to_file_act_values=(),
                 actor_type='simple', critic_type='simple', nan_guard=False, 
                 *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """
        self._actor_type = actor_type
        self._critic_type = critic_type

        self._buffer_args = {}
        for key, value in kwargs.items():
            if key.startswith('buffer.'):
                self._buffer_args[key[len('buffer.'):]] = value

        self._actor_args = {}
        for key, value in kwargs.items():
            if key.startswith('actor.'):
                self._actor_args[key[len('actor.'):]] = value

        self._critic_args = {}
        for key, value in kwargs.items():
            if key.startswith('critic.'):
                self._critic_args[key[len('critic.'):]] = value

        self._update_blocks = update_blocks
        self._buffer_type=buffer_type

        self._force_log = (
            [v[1:].split(':')[0] for v in log_values if v[0] == '!']
            + [v[1:].split(':')[0] for v in log_to_file_values if v[0] == '!']
        )
        self._force_log_memory = [v[1:].split(':')[0] for v in log_memory_values if v[0] == '!']

        super().__init__(observations_space, actions_space, *args, **kwargs)

        _centered = lambda x: x - th.mean(x)
        _batchfirst = lambda x: x[:,0]
        _finite0 = lambda x: th.where(th.isfinite(x), x, 0)
        _first = lambda x: x[0]
        _second = lambda x: x[1]

        self.LOG_GATHER = {
            'mean': th.mean,
            'std': th.std,
            'min': th.min,
            'max': th.max,
            'median': th.median,
            'q1': functools.partial(th.quantile, q=25),
            'q3': functools.partial(th.quantile, q=75),
            'square': th.square,
            'abs': th.abs,
            'log': th.log,
            'centered': _centered,
            'batchfirst': _batchfirst,
            'finite0': _finite0,
            'first': _first,
            'second': _second,
        }
        for i in range(20):
            def get_i(x, i):
                return x[..., i]
            self.LOG_GATHER[str(i)] = functools.partial(get_i, i=i)

        def gather(x, funs):
            for fun in funs:
                x = fun(x.to(th.float32))
            return x

        def prepare_log_values(spec):
            target_list = []
            for log_value in spec:
                vg = log_value.lstrip('!').split(':')
                val = vg[0]
                if len(vg) > 1:
                    funs = [self.LOG_GATHER[fun] for fun in reversed(vg[1].split('_'))]
                    _gather = functools.partial(gather, funs=funs)
                else:
                    _gather = lambda x: x

                target_list.append((log_value, val, _gather))
            return target_list

        def check_log_values(log_values, call_list_data, call_list, where=""):
            for _, value, _ in log_values:
                assert value in call_list_data or value in [n for n, _, _ in call_list],\
                    f'Error: {value} not calculated {where}'


        self._log_values = prepare_log_values(log_values)
        self._log_to_file_values = prepare_log_values(log_to_file_values)
        self._log_memory_values = prepare_log_values(log_memory_values)
        self._log_to_file_memory_values = prepare_log_values(log_to_file_memory_values)
        self._log_act_values = prepare_log_values(log_act_values)
        self._log_to_file_act_values = prepare_log_values(log_to_file_act_values)
        self._nan_guard = nan_guard
        self._nan_log_prev_mem_batch = None
        self._nan_log_prev_batch = None

        self._init_log_act_automodel()
        check_log_values(self._log_values, self._call_list_data, self._call_list)
        check_log_values(self._log_to_file_values, self._call_list_data, self._call_list)
        if self._memory_call_list:
            check_log_values(self._log_memory_values, self._memory_call_list_data, self._memory_call_list, "in memory updates")
            check_log_values(self._log_to_file_memory_values, self._memory_call_list_data, self._memory_call_list, "in memory updates")

    def _init_automodel(self, skip=()):
        self.register_method("mask", self._calculate_mask, {"lengths": "lengths", "n": "n"})
        self.register_method('time_step', lambda: self._th_time_step, {})
        self.register_method('gamma', lambda: self._gamma, {})

        self.register_component('actor', self._actor)
        self.register_component('critic', self._critic)
        self.register_component('memory', self._memory)
        self.register_component('base', self)

        self._init_automodel_overrides()

        self._call_list, self._call_list_data = self.prepare_default_call_list(self.DATA_FIELDS, additional=self._force_log)

        print(*[x[0] for x in self._call_list], sep='\n')

        if self._memory.priority:
            self.register_method('memory_priority', *self._memory.priority)
            self._memory_call_list, self._memory_call_list_data = self.prepare_call_list(
                ['base.memory_priority'] + self._force_log_memory, self.DATA_FIELDS)

            self._buffer_update_loader = self._get_experience_replay_generator(seq=True, fields=self._memory_call_list_data)
        else:
            self._memory_call_list = self._memory_call_list_data = None

    def _init_log_act_automodel(self):
        if self._log_act_values or self._log_to_file_act_values:
            to_log = list({v for _, v, _ in self._log_act_values} | {v for _, v, _ in self._log_to_file_act_values})
            self._log_act_call_list, self._log_act_call_list_data = self.prepare_call_list(to_log, self.ACT_DATA_FIELDS)
        else:
            self._log_act_call_list = self._log_act_call_list_data = None

    def _init_automodel_overrides(self) -> None:
        pass

    def _init_actor(self) -> BaseActor:
        return self.ACTORS[self._actor_type][self._is_discrete](
            self._observations_space, self._actions_space, th_time_step=self._th_time_step,
            batch_size=self._batch_size, num_parallel_envs=self._num_parallel_envs, device=self.device, **self._actor_args
        )

    def _init_data_loader(self, _) -> None:
        self._data_loader = self._get_experience_replay_generator(fields=self._call_list_data)

    def _init_replay_buffer(self, _, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            self._actions_shape = (1, )
        else:
            self._actions_shape = self._actions_space.shape

        buffer_cls, buffer_base_args = self.BUFFERS[self._buffer_type]

        self._memory = buffer_cls(
            action_spec=BufferFieldSpec(shape=self._actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            policy_spec=policy_spec,
            num_buffers=self._num_parallel_envs,
            device=self.device,
            **buffer_base_args,
            **self._buffer_args
        )

    def _calculate_mask(self, lengths, n):
        return (th.arange(n, device=lengths.device) < th.unsqueeze(lengths, 1)).to(th.float32)

    def _init_critic(self) -> Critic:
        # if self._is_obs_discrete:
        #     return TabularCritic(self._observations_space, None, self._tf_time_step)
        # else:
        return self.CRITICS[self._critic_type](self._observations_space, th_time_step=self._th_time_step, device=self.device, **self._critic_args)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        logged = [np.nan for _ in self._log_to_file_memory_values]
        if self._time_step > self._learning_starts:
            if self._memory.should_update_block(self._time_step):
                for batch in self._buffer_update_loader(self._update_blocks):
                    data = {f: d for f, d in zip(self._memory_call_list_data, batch)}
                    priorities, logged = self._calculate_memory_update(data)
                    priorities = priorities.detach().cpu().numpy()
                    self._memory.update_block(priorities)
                    if self._nan_guard:
                        if not np.isfinite(priorities).all():
                            print('NaN on memory update')
                            print('Last mem batch:')
                            print_batch(self._nan_log_prev_mem_batch)
                            print('Last batch:')
                            print_batch(self._nan_log_prev_batch)
                        self._nan_log_prev_mem_batch = data

            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])

            outs = []
            for batch in self._data_loader(experience_replay_iterations):
                data = {f: d for f, d in zip(self._call_list_data, batch)}
                out = self._learn_from_experience_batch(data)

                if self._nan_guard:
                    obs = th.as_tensor(data['obs'], device=self.device)
                    if (
                        not np.isfinite(self._actor._forward(obs).detach().cpu().numpy()).all()
                        or not all([np.isfinite(e.detach().cpu().numpy()).all() for e in self._actor._extras_forward(obs)])
                        or not all([np.isfinite(o.detach().cpu().numpy()).all() for o in out])
                    ):
                        print('NaN on learn step')
                        print('Last mem batch:')
                        print_batch(self._nan_log_prev_mem_batch)
                        print('Last batch:')
                        print_batch(self._nan_log_prev_batch)
                    self._nan_log_prev_batch = data
                outs.append([o.detach().cpu().numpy() for o in out])
            return np.concatenate([logged, np.mean(outs, 0)])
    
    def _calculate_memory_update(self, data):
        data = self.call_list(self._memory_call_list, self.as_tensors(data, ignore=self.CONTROL_FIELDS))

        if self.summary_writer is not None:
            for name, value, gather in self._log_memory_values:
                self.summary_writer.add_scalar('memory_update_log/' + name, gather(data[value]), self._th_time_step)

        logged = [gather(data[value]) for _, value, gather in self._log_to_file_memory_values]
        return data['base.memory_priority'].detach(), logged

    def _learn_from_experience_batch(self, data):
        data = self.call_list(self._call_list, self.as_tensors(data, ignore=self.CONTROL_FIELDS))

        if self.summary_writer is not None:
            for name, value, gather in self._log_values:
                self.summary_writer.add_scalar('log/' + name, gather(data[value]), self._th_time_step)
        return tuple(gather(data[value]).detach() for _, value, gather in self._log_to_file_values)

    def predict_action_log(self, observations, action):
        if self._log_act_call_list is not None:
            data = {}
            if 'actions' in self._log_act_call_list_data:
                data['actions'] = action
            if 'obs' in self._log_act_call_list_data:
                data['obs'] = observations
            data = self.call_list(self._log_act_call_list, self.as_tensors(data, ignore=self.CONTROL_FIELDS))

            if self.summary_writer is not None:
                for name, value, gather in self._log_memory_values:
                    self.summary_writer.add_scalar('act_log/' + name, gather(data[value]), self._th_time_step)
            return [gather(data[value]).detach() for _, value, gather in self._log_to_file_act_values]

    def predict_action(self, observations: np.array, is_deterministic: bool = False):
        if observations.dtype == np.float64:
            observations = np.cast[np.float32](observations)

        action, policy, _ = super().predict_action(observations, is_deterministic)
        if not is_deterministic:
            log = self.predict_action_log(observations, action)
            if log is not None:
                log = [v.detach().cpu().numpy() for v in log]
        else:
            log = None
        return action, policy, log

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        return self._memory.get_vec(self._batches_per_env, self._memory.n)

    def _prepare_generator_fields(self, size):
        batch_size_ids = np.arange(size)

        specs = {  # replacing identity functions with none and not calling gives a slight speed increase
            'lengths': ('lengths', None),  # lambda x, lens: x),
            'obs': ('observations', None),  # lambda x, lens: x),
            'obs_next': ('next_observations', lambda x, lens: x[batch_size_ids, lens - 1]),
            'actions': ('actions', None),  # lambda x, lens: x),
            'policies': ('policies', None),  # lambda x, lens: x),
            'rewards': ('rewards', None),  # lambda x, lens: x),
            'dones': ('dones', lambda x, lens: x[batch_size_ids, lens - 1]),
            'priorities': ('priors', lambda x, lens: x[:, 0]),
            'time': ('time', None),  # lambda x, lens: x),
            'batch_size': ('observations', lambda x, lens: x.shape[0]),
            'n': ('observations', lambda x, lens: x.shape[1]),
        }

        dtypes = {
            'lengths': np.int32,
            'obs': np.int32 if self._is_obs_discrete else np.float32,
            'obs_next': np.int32 if self._is_obs_discrete else np.float32,
            'actions': self._actor.action_dtype_np,
            'policies': np.float32,
            'rewards': np.float32,
            'dones': np.bool_,
            'priorities': np.float32,
            'time': np.int32,
            'batch_size': int,
            'n': int,
        }

        return specs, dtypes

    def _get_experience_replay_generator(
            self, seq=False, fields=None):
        if fields is None:
            fields = self.DATA_FIELDS

        specs, dtypes = self._prepare_generator_fields(self._memory.block if seq else self._batch_size)

        field_specs = [specs[f] for f in fields]
        field_dtypes = tuple(dtypes[f] for f in fields)

        def experience_replay_generator(n):
            for _ in range(n):
                batch, lens = self._memory.get_next_block_to_update() if seq else self._fetch_offline_batch()
                batch["lengths"] = lens
                data = tuple(
                    np.cast[dtype](value) if dtype not in (int, float) else value
                    for value, dtype in (
                        (preproc(batch[field], lens) if preproc else batch[field], dtype)
                        for dtype, (field, preproc) in zip(field_dtypes, field_specs)))
                yield data

        return experience_replay_generator

