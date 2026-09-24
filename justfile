# Driving World Model — development task runner.
# Run `just` (or `just --list`) to see all available commands.

default:
    @just --list

# Install all dependencies, including dev tools (ruff, pytest).
install:
    uv sync --group dev

# Run the test suite.
test *ARGS:
    uv run pytest tests/ -v {{ ARGS }}

# Run the test suite with a coverage report.
coverage:
    uv run pytest tests/ --cov=driving_world_model --cov-report=term-missing --cov-report=html

# Lint the codebase.
lint:
    uv run ruff check .

# Auto-fix lint issues where possible.
lint-fix:
    uv run ruff check --fix .

# Check code formatting without modifying files.
format-check:
    uv run ruff format --check .

# Format the codebase in place.
format:
    uv run ruff format .

# Run every check CI runs: lint, format check, and tests.
check: lint format-check test

# Launch single-process training locally (see README for multi-GPU/multi-node).
train *ARGS:
    uv run torchrun --nproc_per_node=1 -m driving_world_model.train {{ ARGS }}

# Remove caches and build artifacts.
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
    find . -type d -name "__pycache__" -exec rm -rf {} +
