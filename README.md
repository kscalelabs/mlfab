<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>


<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/dpshai/mlfab/blob/master/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/mlfab/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/mlfab/actions/workflows/test.yml)

</div>

<br />

# mlfab

## What is this?

This is a framework for trying out machine learning ideas.

## Getting Started

Install the package using:

```bash
pip install mlfab
```

Or, to install the latest branch:

```bash
pip install 'mlfab @ git+https://github.com/kscalelabs/mlfab.git@master'
```

### Simple Example

This framework provides an abstraction for quickly implementing and training PyTorch models. The workhorse for doing this is `mlfab.Task`, which wraps all of the training logic into a single cohesive unit. We can override functions on that method to get special functionality, but the default functionality is often good enough. Here's an example for training an MNIST model:

```python

from dataclasses import dataclass

import torch.nn.functional as F
from dpshdl.dataset import Dataset
from dpshdl.impl.mnist import MNIST
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

import mlfab


@dataclass
class Config(mlfab.Config):
    in_dim: int = mlfab.field(1, help="Number of input dimensions")


class MnistClassification(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.model = nn.Sequential(
            nn.Conv2d(config.in_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def set_loggers(self) -> None:
        self.add_logger(
            mlfab.StdoutLogger(),
            mlfab.TensorboardLogger(self.exp_dir),
        )

    def get_dataset(self, phase: mlfab.Phase) -> Dataset[tuple[Tensor, Tensor]]:
        root_dir = mlfab.get_data_dir() / "mnist"
        return MNIST(root_dir=root_dir, train=phase == "train")

    def build_optimizer(self) -> Optimizer:
        return mlfab.Adam.get(self, lr=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        x, y = batch
        yhat = self(x)
        self.log_step(batch, yhat, state)
        return F.cross_entropy(yhat, y.squeeze(-1))

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        (x, y), yhat = batch, output

        def get_label_strings() -> list[str]:
            ytrue, ypred = y.squeeze(-1), yhat.argmax(-1)
            return [f"ytrue={ytrue[i]}, ypred={ypred[i]}" for i in range(len(ytrue))]

        self.log_labeled_images("images", lambda: (x, get_label_strings()))


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config(batch_size=16))
```

Let's break down each part individually.

### Config

Tasks are parametrized using a config dataclass. The `ml.field` function is a lightweight wrapper around `dataclasses.field` which is a bit more ergonomic, and `ml.Config` is a bigger dataclass which contains a bunch of other options for configuring training.

```python
@dataclass
class Config(mlfab.Config):
    in_dim: int = mlfab.field(1, help="Number of input dimensions")
```

### Model

All tasks should subclass `ml.Task` and override the generic `Config` with the task-specific config. This is very important, not just because it makes your life easier by working nicely with your typechecker, but because the framework looks at the generic type when resolving the config for the given task.

```python
class MnistClassification(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.model = nn.Sequential(
            nn.Conv2d(config.in_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
```

### Loggers

`mlfab` supports logging to multiple downstream loggers, and provides a bunch of helper functions for doing common logging operations, like rate limiting, converting image resolution to normal sizes, overlaying captions on images, and more.

If this function is not overridden, the task will just log to `stdout`.

```python
def set_loggers(self) -> None:
    self.add_logger(
        mlfab.StdoutLogger(),
        mlfab.TensorboardLogger(self.exp_dir),
    )
```

### Datasets

The task should return the dataset used for training, based on the phase. `ml.Phase` is a string literal with values in `["train", "valid", "test"]`. `mlfab.get_data_dir()` returns the data directory, which can be set in a configuration file which lives in `~/.mlfab.yml`. The default configuration file will be written on first run if it doesn't exist yet.

```python
def get_dataset(self, phase: mlfab.Phase) -> Dataset[tuple[Tensor, Tensor]]:
    root_dir = mlfab.get_data_dir() / "mnist"
    return MNIST(root_dir=root_dir, train=phase == "train")
```

### Optimizers

```python
def build_optimizer(self) -> Optimizer:
    return mlfab.Adam.get(self, lr=1e-3)
```

### Compute Loss

Each `mlfab` model should either implement the `forward` function, which should take a batch from the dataset and return the loss, or, if more control is desired, the `get_loss` function can be overridden.

```python
def forward(self, x: Tensor) -> Tensor:
    return self.model(x)

def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
    x, y = batch
    yhat = self(x)
    self.log_step(batch, yhat, state)
    return F.cross_entropy(yhat, y.squeeze(-1))
```

### Logging

When we call `log_step` in the `get_loss` function, it delegates to either `log_train_step`, `log_valid_step` or `log_test_step`, depending on what `state.phase` is. In this case, on each validation step we log images of the MNIST digits with the labels that our model predicts.

```python
def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
    (x, y), yhat = batch, output

    def get_label_strings() -> list[str]:
        ytrue, ypred = y.squeeze(-1), yhat.argmax(-1)
        return [f"ytrue={ytrue[i]}, ypred={ypred[i]}" for i in range(len(ytrue))]

    self.log_labeled_images("images", lambda: (x, get_label_strings()))
```

### Running

We can launch a training job using the `launch` class method. The config can be a `Config` object, or it can be the path to a `config.yaml` file located in the same directory as the task file. You can additionally provide the `launcher` argument, which supports training the model across multiple GPUs or nodes.

```python
if __name__ == "__main__":
    MnistClassification.launch(Config(batch_size=16))
```
