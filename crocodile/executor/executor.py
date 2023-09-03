from dataclasses import dataclass
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ExecutorConfig:
    ...


class Executor(ABC):
    __name__: str

    @abstractmethod
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        pass


@dataclass
class LocalExecutorConfig(ExecutorConfig):
    ...


class LocalExecutor(Executor):
    __name__ = "local"

    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        return func(*args, **kwargs)
