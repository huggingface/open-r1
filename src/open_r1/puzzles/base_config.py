from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseConfig:
    num_tasks: int = 100
    seed: int | None = None
