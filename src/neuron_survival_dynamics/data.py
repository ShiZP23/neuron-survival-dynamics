import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DatasetBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


TASKS = ["simple", "medium", "hard"]


def _target_function(task: str, x: np.ndarray) -> np.ndarray:
    x0 = x[:, 0]
    x1 = x[:, 1]

    if task == "simple":
        return x0 + x1
    if task == "medium":
        return np.sin(2.0 * math.pi * x0) + 0.3 * (x1 ** 2) + 0.2 * x0 * x1
    if task == "hard":
        return np.sin(4.0 * math.pi * x0) * np.cos(2.0 * math.pi * x1) + 0.2 * (x0 ** 3)

    raise ValueError(f"Unknown task: {task}")


def make_dataset(
    task: str,
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    noise: float = 0.0,
    seed: int = 0,
) -> DatasetBundle:
    if task not in TASKS:
        raise ValueError(f"Task must be one of {TASKS}")

    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1.0, 1.0, size=(n_train, 2))
    x_val = rng.uniform(-1.0, 1.0, size=(n_val, 2))
    x_test = rng.uniform(-1.0, 1.0, size=(n_test, 2))

    y_train = _target_function(task, x_train)
    y_val = _target_function(task, x_val)
    y_test = _target_function(task, x_test)

    if noise > 0.0:
        y_train = y_train + rng.normal(scale=noise, size=y_train.shape)
        y_val = y_val + rng.normal(scale=noise, size=y_val.shape)
        y_test = y_test + rng.normal(scale=noise, size=y_test.shape)

    return DatasetBundle(
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train[:, None], dtype=torch.float32),
        x_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val[:, None], dtype=torch.float32),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test[:, None], dtype=torch.float32),
    )


def make_grid(n: int = 200) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, n)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return grid


def evaluate_target(task: str, x: torch.Tensor) -> torch.Tensor:
    np_x = x.detach().cpu().numpy()
    y = _target_function(task, np_x)
    return torch.tensor(y[:, None], dtype=torch.float32)
