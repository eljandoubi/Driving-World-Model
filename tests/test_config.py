"""Tests for TrainingConfig validation."""

import pytest

from driving_world_model.config import TrainingConfig


def test_n_gpus_zero_raises():
    with pytest.raises(AssertionError, match="n_gpus must be > 0"):
        TrainingConfig(n_gpus=0)


def test_n_nodes_zero_raises():
    with pytest.raises(AssertionError, match="n_nodes must be > 0"):
        TrainingConfig(n_nodes=0)


def test_checkpoint_every_not_multiple_of_log_every_raises():
    with pytest.raises(AssertionError, match="checkpoint_every must be a multiple"):
        TrainingConfig(log_every=300, checkpoint_every=1000)


def test_invalid_dropout_raises():
    with pytest.raises(AssertionError, match="dropout must be in"):
        TrainingConfig(dropout=1.0)


def test_invalid_activation_raises():
    with pytest.raises(AssertionError, match="activation must be one of"):
        TrainingConfig(activation="relu")  # pyright: ignore[reportArgumentType]


def test_invalid_loss_type_raises():
    with pytest.raises(AssertionError, match="loss_type must be"):
        TrainingConfig(loss_type="huber")  # pyright: ignore[reportArgumentType]


def test_resume_missing_file_raises():
    with pytest.raises(AssertionError, match="does not exist"):
        TrainingConfig(resume="/nonexistent/path/to/checkpoint.pt")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("num_tokens", 0, "num_tokens must be > 0"),
        ("hidden_dim", 0, "hidden_dim must be > 0"),
        ("learning_rate", 0, "learning_rate must be > 0"),
        ("epochs", 0, "epochs must be > 0"),
        ("max_grad_norm", 0, "max_grad_norm must be > 0"),
        ("batch_size", 0, "batch_size must be > 0"),
        ("log_every", 0, "log_every must be > 0"),
        ("image_size", 0, "image_size must be > 0"),
        ("patience", 0, "patience must be > 0"),
        ("min_delta", -1.0, "min_delta must be >= 0"),
    ],
)
def test_non_positive_fields_raise(field, value, match):
    with pytest.raises(AssertionError, match=match):
        TrainingConfig(**{field: value})


def test_log_every_and_checkpoint_every_scaled_by_world_size_and_batch():
    config = TrainingConfig(log_every=100, checkpoint_every=1000, batch_size=4, n_gpus=2)
    assert config.log_every == 13
    assert config.checkpoint_every == 125


def test_update_paths_without_run_id_raises():
    config = TrainingConfig()
    with pytest.raises(AssertionError, match="run_id must be set"):
        config.update_paths()


def test_set_id_and_update_paths(tmp_path):
    config = TrainingConfig(runs_dir=tmp_path)
    config.set_id("run123")
    assert config.run_id == "run123"

    config.update_paths()

    assert config.run_dir == tmp_path / "run123"
    assert config.checkpoint_dir == tmp_path / "run123" / "checkpoints"
    assert config.plot_dir == tmp_path / "run123" / "plots"
    assert config.checkpoint_dir.is_dir()
    assert config.plot_dir.is_dir()
