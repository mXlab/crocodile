from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Any


@dataclass
class BaseExecutorConfig:
    pass


class Executor(ABC):
    @abstractmethod
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        pass


@dataclass
class LocalExecutorConfig(BaseExecutorConfig):
    pass


class LocalExecutor(Executor):
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        return func(*args, **kwargs)
