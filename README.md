# Driving World Model

[![CI](https://github.com/eljandoubi/driving-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/eljandoubi/driving-world-model/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple)](https://github.com/astral-sh/uv)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A world model for autonomous driving that learns to predict the next visual observation given the current frame and a driving action. The model is designed for hyperparameter tuning — all training parameters are exposed as CLI flags, making it straightforward to run sweeps with tools like W&B Sweeps, Optuna, or grid search scripts.

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Default Configuration](#default-configuration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Development](#development)
- [Project Structure](#project-structure)
- [License](#license)

## Motivation

World models enable planning and simulation without interacting with the real environment. This project frames next-frame prediction as a conditional generation task: given the current camera frame $x_t$ and a 3-dimensional action $a_t = (\text{throttle}, \text{steer}, \text{brake})$, the model predicts $x_{t+1}$.

## Architecture

| Component | Description |
|-----------|-------------|
| **WorldModel** | Pretrained Stable Diffusion UNet conditioned on actions via cross-attention |
| **ActionEmbedder** | Feed-forward network (SwiGLU) projecting $\mathbb{R}^3 \to \mathbb{R}^{T \times D}$ multi-token embeddings |
| **Loss** | L1 (default) or L2 between predicted and ground-truth residual frame |

The UNet backbone is loaded from `runwayml/stable-diffusion-v1-5` (configurable). Actions are embedded into `num_tokens` vectors of dimension `cross_attention_dim` and injected as the encoder hidden states.

## Dataset

Uses the [CARLA Autopilot Multimodal Dataset](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) streamed via HuggingFace `datasets`. Each sample pairs consecutive frames within the same driving run: $(x_t, a_t) \to x_{t+1}$.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.12+.

```bash
uv sync                 # production dependencies
uv sync --group dev     # + ruff, pytest, pytest-cov for development
```

The project installs as the `driving_world_model` package (src layout), so every module is imported as `driving_world_model.<module>` and every entry point is run with `python -m driving_world_model.<module>`.

## Training

`train.py` always launches through `torchrun` — even for a single process — since it reads its rank/world-size from the environment `torchrun` sets up (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`).

```bash
uv run torchrun --nproc_per_node=1 -m driving_world_model.train [OPTIONS]
```

or, equivalently:

```bash
just train [OPTIONS]
```

All fields in `TrainingConfig` are valid CLI flags (via HuggingFace `HfArgumentParser`). This makes it trivial to run hyperparameter sweeps:

```bash
# Example: sweep over learning rate and scheduler
uv run torchrun --nproc_per_node=1 -m driving_world_model.train --learning_rate 3e-4 --scheduler_t0 5 --scheduler_t_mult 1
uv run torchrun --nproc_per_node=1 -m driving_world_model.train --learning_rate 1e-4 --scheduler_t0 20 --scheduler_t_mult 2
```

### Resume from checkpoint

```bash
uv run torchrun --nproc_per_node=1 -m driving_world_model.train \
    --resume runs/<run_id>/checkpoints/best_checkpoint.pt --run_id <run_id>
```

### Distributed Training

Training supports DistributedDataParallel (DDP) for multi-GPU and multi-node setups. Data is sharded across ranks, gradients are synchronized automatically, and only rank 0 handles logging, checkpointing, and video generation.

`torchrun` is the only supported launcher — it owns rank/world-size assignment via environment variables, so the config's `--n_gpus`/`--n_nodes` flags don't launch anything by themselves. Set them to match the topology you pass to `torchrun` so `log_every`/`checkpoint_every` are scaled correctly for the true global batch size.

#### Single-node multi-GPU

```bash
uv run torchrun --nproc_per_node=4 -m driving_world_model.train --n_gpus 4
```

#### Multi-node

Each machine runs `torchrun` independently. `--node_rank` is a `torchrun` flag (not a `TrainingConfig` field), and `--master_addr` should point to node 0 so all nodes can rendezvous.

```bash
# Node 0 (master):
uv run torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=12355 \
    -m driving_world_model.train --n_gpus 4 --n_nodes 2

# Node 1:
uv run torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=12355 \
    -m driving_world_model.train --n_gpus 4 --n_nodes 2
```

## Default Configuration

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_name` | `runwayml/stable-diffusion-v1-5` | Pretrained UNet backbone |
| `num_tokens` | `16` | Number of action embedding tokens |
| `hidden_dim` | `2048` | Action embedder hidden dimension |
| `activation` | `swiglu` | Activation in action embedder (`gelu`, `geglu`, `swiglu`, etc.) |
| `loss_type` | `l1` | Loss function (`l1` or `l2`) |
| `dropout` | `0.0` | Dropout in action embedder |

### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | AdamW learning rate |
| `max_grad_norm` | `1.0` | Gradient clipping max norm |
| `scheduler_t0` | `10` | CosineAnnealingWarmRestarts $T_0$ (epochs per first restart) |
| `scheduler_t_mult` | `2` | CosineAnnealingWarmRestarts $T_{mult}$ (period multiplier) |
| `epochs` | `100000` | Maximum training epochs |

### Early Stopping & Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | `10` | Early stopping patience (number of validations without improvement) |
| `min_delta` | `1e-8` | Minimum improvement to reset patience counter |
| `log_every` | `1000` | Log training loss every N steps |
| `checkpoint_every` | `10000` | Validate & checkpoint every N steps |

### Distributed

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gpus` | `1` | GPUs per node |
| `n_nodes` | `1` | Number of nodes (machines) |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | `256` | Input image resize resolution |
| `batch_size` | `2` | Training batch size (per GPU) |
| `buffer_size` | `1000` | Prefetch buffer size for streaming dataset |
| `pin_memory` | `True` | Pin memory for GPU transfer |

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `runs_dir` | `runs` | Base directory for all run outputs |
| `checkpoint_dir` | `checkpoints` | Subdirectory for checkpoints (under `runs/<run_id>/`) |
| `plot_dir` | `plots` | Subdirectory for plots (under `runs/<run_id>/`) |
| `resume` | `""` | Path to checkpoint to resume from |
| `run_id` | `None` | W&B run ID (auto-generated if not set) |

## Hyperparameter Tuning

Key parameters to sweep:

- **`learning_rate`**: Try log-uniform in `[1e-5, 1e-3]`
- **`num_tokens`**: `{4, 8, 16}` — controls action conditioning capacity
- **`hidden_dim`**: `{512, 1024, 2048}` — action embedder width
- **`scheduler_t0`** / **`scheduler_t_mult`**: Controls warm restart frequency
- **`dropout`**: `[0.0, 0.3]`
- **`image_size`**: `{128, 256, 512}` — resolution/compute trade-off
- **`batch_size`**: `{4, 8, 16, 32}` — adjust with available GPU memory

All parameters are logged to W&B automatically, enabling comparison across runs.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management, [Ruff](https://github.com/astral-sh/ruff) for linting/formatting, and [pytest](https://docs.pytest.org/) for testing. A [`justfile`](justfile) wraps the common commands — run `just` to list them all:

```bash
just install       # uv sync --group dev
just test          # run the test suite
just coverage      # run tests with a coverage report
just lint          # ruff check
just format        # ruff format
just check         # lint + format-check + test (what CI runs)
```

Without `just`, the equivalent `uv run` commands work the same way, e.g. `uv run pytest tests/ -v`.

Every push and pull request to `main` runs the [CI workflow](.github/workflows/ci.yml), which lints, format-checks, and tests the project on Python 3.12.

## Project Structure

```
├── src/
│   └── driving_world_model/
│       ├── model.py            # WorldModel + ActionEmbedder
│       ├── train.py            # Training loop with scheduler, early stopping, checkpointing
│       ├── validate.py         # Validation/test evaluation
│       ├── dataset.py          # Streaming dataset from HuggingFace
│       ├── config.py           # TrainingConfig dataclass (all hyperparameters)
│       ├── checkpoint.py       # Save/load checkpoints
│       ├── early_stopping.py   # EarlyStopping utility
│       ├── logger.py           # Rank-aware logging with tqdm integration
│       └── plot.py             # Video generation for prediction visualization
├── tests/                      # Unit tests (mirrors src/driving_world_model)
├── .github/workflows/ci.yml    # Lint + test CI pipeline
├── justfile                    # Dev task runner (install, test, lint, format, train)
└── pyproject.toml              # Dependencies, build system & tool config
```

## License

Licensed under the [Apache License 2.0](LICENSE).