from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ExecutorConfig:
    ...


class Executor(ABC):
    @abstractmethod
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        pass


@dataclass
class LocalExecutorConfig(ExecutorConfig):
    ...


class LocalExecutor(Executor):
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        return func(*args, **kwargs)
