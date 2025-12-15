import csv
from abc import ABC, abstractmethod
import datetime
from collections import defaultdict, OrderedDict
import logging
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Any


class Logger(ABC):
    def __init__(self, keys: List[str]):
        """Collects key-values logs

        Args:
            keys: list of keys to be logged
        """
        self._data = defaultdict(dict)
        self._keys = keys + ['timestamp']

    def log_values(self, key_values: Dict[str, Any]):
        """Stores single row. Only values defined under keys passed in the constructor method
        are logged.

        Args:
            key_values: dictionary with values to store
        """
        stored_row = OrderedDict()
        for key in self._keys:
            stored_row[key] = key_values.get(key)

        stored_row['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._store_log(stored_row)

    @abstractmethod
    def _store_log(self, key_values: Dict[str, Any]):
        ...


class CSVLogger(Logger):

    def __init__(self, file_path: Path, *args, **kwargs):
        """Logs data into csv file. IMPORTANT: close() method have to be called at the end of the run.

        Args:
            file_path: path to the log file
        """
        super().__init__(*args, **kwargs)
        self._file = open(str(file_path), 'wt')
        self._file_path = file_path
        self._writer = csv.writer(self._file, delimiter=',')
        self._writer.writerow(self._keys)

    def _store_log(self, key_values: OrderedDict):
        row = list(key_values.values())
        self._writer.writerow(row)

    def dump(self):
        self._file.flush()

    def close(self):
        self._file.close()


class ConsoleLogger(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def info(self, message):
        ...

    @abstractmethod
    def episode_finish(self, step, episode, rewards):
        ...

    @abstractmethod
    def evaluation_results(self, step, rewards):
        ...

    @abstractmethod
    def timestep(self, step):
        ...

    @abstractmethod
    def error(self):
        ...

    @abstractmethod
    def flush(self, step):
        ...

    @abstractmethod
    def log_values(self, values):
        ...


class DefaultConsoleLogger(ConsoleLogger):
    def __init__(self) -> None:
        super().__init__()
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        )

    def info(self, message):
        logging.info(message)

    def episode_finish(self, step, episode, rewards):
        logging.info(f"finished episode {episode}, "
                f"return: {rewards}, "
                f"total time steps done: {step}")

    def evaluation_results(self, step, rewards):
        for reward in rewards:
            logging.info(f"evaluation run, "
                f"return: {reward}")

    def timestep(self, step):
        pass

    def error(self):
        traceback.print_exc()

    def flush(self, step):
        pass

    def log_values(self, values):
        logging.info(f'Logged values to file: {values}')


class PeriodicConsoleLogger(ConsoleLogger):
    def __init__(self, n) -> None:
        super().__init__()
        
        self._n = n
        self._next_log = 0
        self._infos = []
        self._episodes_finished = []
        self._evaluations = []
        self._values_logged = []

    def info(self, message):
        self._infos.append(message)

    def episode_finish(self, step, episode, rewards):
        self._episodes_finished.append((step, episode, rewards))

    def evaluation_results(self, step, rewards):
        self._evaluations.append((step, rewards))

    def timestep(self, step):
        if step < self._next_log:
            return
        
        self._print(step)
        self._next_log += self._n

    def _print(self, step):
        print(f'Step: {step}')
        if self._infos:
            print(f'Infos: {"; ".join(self._infos)}')
        if self._episodes_finished:
            print(f'Episodes: {"; ".join(" ".join([str(s), str(e), str(r)]) for s, e, r in self._episodes_finished)}')
        if self._evaluations:
            print(f'Evaluations: {"; ".join(" ".join([str(r) for r in rewards[1]]) for rewards in self._evaluations)}')
        if self._values_logged:
            print(f'Values logged: "[{";".join(",".join(f"{k}:{v}" for k, v in vals.items()) for vals in self._values_logged)}"]"')
        sys.stdout.flush()

        self._infos = []
        self._episodes_finished = []
        self._evaluations = []
        self._values_logged = []

    def error(self):
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(f'Error type: {exc_type.__name__}({exc_value})')
        while exc_tb.tb_next and 'site-packages' not in exc_tb.tb_next.tb_frame.f_code.co_filename:
            exc_tb = exc_tb.tb_next
        print(f'Error function: {exc_tb.tb_frame.f_code.co_name} in {exc_tb.tb_frame.f_code.co_filename} ({exc_tb.tb_frame.f_lineno})')
        traceback.print_exc()
        sys.stdout.flush()

    def flush(self, step):
        self._print(step)

    def log_values(self, values):
        self._values_logged.append(values)

