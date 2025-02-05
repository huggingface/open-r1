from __future__ import annotations

from abc import ABC, abstractmethod
from random import randint

import numpy as np

from .base_config import BaseConfig


MAX_INT = np.iinfo(np.int64).max


class BaseTask(ABC):
    config_class = None

    def __init__(self, config: BaseConfig = None):
        if config is not None:
            self.config = config
        elif self.config_class is not None:
            self.config = self.config_class()  # Instantiate the default config for this task
        else:
            raise ValueError("No config provided and no default config_class set for this task")

        # We generate individual sample rngs using seed + idx, so we scramble the seeds to large ints first
        # to avoid sample overlap between datasets using common similar, small seeds (like 0 and 42 and 123)
        self.seed = self.config.seed or randint(0, MAX_INT)
        seed_scrambler = np.random.default_rng(self.seed)
        self.scrambled_seed = seed_scrambler.integers(low=0, high=MAX_INT, size=None)

    def __len__(self):
        return self.config.num_tasks

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_rng(self, idx) -> np.random.Generator:
        return np.random.default_rng(self.scrambled_seed + idx)

    def __getitem__(self, item) -> tuple:
        rng = self.get_rng(item)
        return self.generate_sample(self.config, rng)

    @abstractmethod
    def generate_sample(self, config: BaseConfig, rng: np.random.Generator) -> tuple:
        # This should return a tuple of (output, answer)
        raise NotImplementedError

    @abstractmethod
    def verify(self, output, answer) -> float:
        # This should return a score between 0. and 1. based on how well the output matches the answer
        raise NotImplementedError
