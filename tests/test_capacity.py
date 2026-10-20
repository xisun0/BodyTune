import numpy as np
import pytest
import torch

from roboscore.capacity import reevaluate, train
from roboscore.models import DiameterPerformer, ModelConfig
from roboscore.train import dataset


@pytest.mark.parametrize("hidden", [64, 128, 256])
def test_exact_diameter_policy_state_and_bounds(hidden):
    torch.set_num_threads(2)
    model = DiameterPerformer(ModelConfig(hidden=hidden, tokens=1, steps=12))
    q0 = torch.tensor([[0.3, 1.2], [-0.4, -1.4]])
    d = torch.tensor([0.3, 0.6], requires_grad=True)
    q = model(d, q0)
    torch.testing.assert_close(q[:, 0], q0)
    assert q.shape == (2, 12, 2)
    assert torch.diff(q, dim=1).abs().max() <= 0.1 + 1e-6
    q.square().mean().backward()
    assert d.grad.abs().sum() > 0
    assert not hasattr(model, "embedding")


def test_capacity_training_reloads_identical_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = ModelConfig(tokens=1)
    for i, name in enumerate(("train", "val", "test")):
        dataset(6, 100 + i, config, "circles", source / f"{name}_demonstrations.npz")
    report = train(source, tmp_path / "run", epochs=1, min_epochs=1, threads=2)
    metrics = reevaluate(tmp_path / "run", source)
    assert metrics == report["summary"]["test"]
    assert report["data"]["counts"] == {"train": 6, "val": 6, "test": 6}
    with np.load(source / "test_demonstrations.npz") as data:
        np.testing.assert_array_equal(
            data["diameter"].astype("float32"),
            np.load(tmp_path / "run/test_rollouts.npz")["diameter"],
        )
    path = source / "train_demonstrations.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hashes changed"):
        reevaluate(tmp_path / "run", source)
