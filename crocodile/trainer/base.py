"""Base trainer class."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TrainConfig:
    """Base trainer class."""


class Trainer(Protocol):
    """Base trainer class."""

    def train(self):
        """Train function of model"""
