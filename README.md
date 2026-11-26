# Driving world model

[![CI](https://github.com/eljandoubi/driving-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/eljandoubi/driving-world-model/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A world model for autonomous driving that learns to predict the next visual observation given the current frame and a driving action. All training parameters are exposed as CLI flags, so sweeps with tools like W&B Sweeps, Optuna or grid search scripts need no code change.

World models enable planning and simulation without interacting with the real environment. This project frames next-frame prediction as a conditional generation task: given the current camera frame $x_t$ and a 3-dimensional action $a_t = (\text{throttle}, \text{steer}, \text{brake})$, the model predicts $x_{t+1}$.

## How to run

### Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.12 or newer.

```bash
uv sync                 # production dependencies
uv sync --group dev     # + ruff, pytest, pytest-cov for development
```

The project installs as the `driving_world_model` package (src layout). Every module is imported as `driving_world_model.<module>`, and every entry point is run with `python -m driving_world_model.<module>`.

### Training

`train.py` always launches through `torchrun`, even for a single process. It reads its rank and world size from the environment that `torchrun` sets up (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`).

```bash
uv run torchrun --nproc_per_node=1 -m driving_world_model.train [OPTIONS]
```

or, equivalently:

```bash
just train [OPTIONS]
```

All fields in `TrainingConfig` are valid CLI flags (via HuggingFace `HfArgumentParser`), so a hyperparameter sweep is a list of commands:

```bash
# Example: sweep over learning rate and scheduler
uv run torchrun --nproc_per_node=1 -m driving_world_model.train --learning_rate 3e-4 --scheduler_t0 5 --scheduler_t_mult 1
uv run torchrun --nproc_per_node=1 -m driving_world_model.train --learning_rate 1e-4 --scheduler_t0 20 --scheduler_t_mult 2
```

### Resume from a checkpoint

```bash
uv run torchrun --nproc_per_node=1 -m driving_world_model.train \
    --resume runs/<run_id>/checkpoints/best_checkpoint.pt --run_id <run_id>
```

### Distributed training

Training supports DistributedDataParallel (DDP) for multi-GPU and multi-node setups. Data is sharded across ranks and gradients are synchronized automatically. Only rank 0 handles logging, checkpointing and video generation.

`torchrun` is the only supported launcher. It owns the rank and world-size assignment through environment variables, so the `--n_gpus` and `--n_nodes` flags of the config launch nothing by themselves. Set them to match the topology you pass to `torchrun`, so that `log_every` and `checkpoint_every` scale correctly for the true global batch size.

Single node, several GPUs:

```bash
uv run torchrun --nproc_per_node=4 -m driving_world_model.train --n_gpus 4
```

Several nodes. Each machine runs `torchrun` on its own. `--node_rank` is a `torchrun` flag, not a `TrainingConfig` field, and `--master_addr` should point to node 0 so that all nodes can rendezvous.

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

## Default configuration

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
| `epochs` | `1000` | Maximum training epochs |

### Early stopping and checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | `10` | Early stopping patience (number of validations without improvement) |
| `min_delta` | `1e-8` | Minimum improvement to reset patience counter |
| `log_every` | `1000` | Log training loss every N steps |
| `checkpoint_every` | `10000` | Validate and checkpoint every N steps |

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

## Design choices

| Component | Description |
|-----------|-------------|
| `WorldModel` | Pretrained Stable Diffusion UNet conditioned on actions via cross-attention |
| `ActionEmbedder` | Feed-forward network (SwiGLU) projecting $\mathbb{R}^3 \to \mathbb{R}^{T \times D}$ multi-token embeddings |
| Loss | L1 (default) or L2 between predicted and ground-truth residual frame |

The UNet backbone is loaded from `runwayml/stable-diffusion-v1-5` (configurable). Actions are embedded into `num_tokens` vectors of dimension `cross_attention_dim` and injected as the encoder hidden states.

The data is the [CARLA Autopilot Multimodal Dataset](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset), streamed through HuggingFace `datasets`. Each sample pairs consecutive frames within the same driving run: $(x_t, a_t) \to x_{t+1}$.

Parameters worth sweeping:

- `learning_rate`: log-uniform in `[1e-5, 1e-3]`
- `num_tokens`: `{4, 8, 16}`, the capacity of the action conditioning
- `hidden_dim`: `{512, 1024, 2048}`, the width of the action embedder
- `scheduler_t0` and `scheduler_t_mult`: the frequency of warm restarts
- `dropout`: `[0.0, 0.3]`
- `image_size`: `{128, 256, 512}`, a resolution and compute trade-off
- `batch_size`: `{4, 8, 16, 32}`, adjusted to the available GPU memory

All parameters are logged to W&B automatically, which makes runs comparable.

## Project structure

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

## Status and limits

- No training results are included yet, and the sweep ranges above are suggestions, not results.

## Development

The project uses [uv](https://github.com/astral-sh/uv) for dependencies, [Ruff](https://github.com/astral-sh/ruff) for linting and formatting, and [pytest](https://docs.pytest.org/) for tests. A [`justfile`](justfile) wraps the common commands. Run `just` to list them all:

```bash
just install       # uv sync --group dev
just test          # run the test suite
just coverage      # run tests with a coverage report
just lint          # ruff check
just format        # ruff format
just check         # lint + format-check + test (what CI runs)
```

Without `just`, the equivalent `uv run` commands work the same way, for example `uv run pytest tests/ -v`.

Every push and pull request to `main` runs the [CI workflow](.github/workflows/ci.yml), which lints, format-checks and tests the project on Python 3.12.

## License

Licensed under the [Apache License 2.0](LICENSE).
