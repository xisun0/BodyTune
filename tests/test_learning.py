import numpy as np
import torch

from roboscore.models import ModelConfig, MotionModel, features, fk
from roboscore.train import train


def test_torch_kinematics_matches_numpy():
    from roboscore.kinematics import Arm

    q = torch.tensor([[[0.2, 1.1], [-0.8, -1.2]]], requires_grad=True)
    np.testing.assert_allclose(fk(q).detach().numpy(), Arm().fk(q.detach().numpy()), atol=2e-7)
    fk(q).sum().backward()
    assert q.grad.abs().sum() > 0


def test_fixed_channel_and_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = ModelConfig(steps=16)
    model = MotionModel(config)
    a = torch.rand(3, 16, 2)
    channel, ids = model.notator(features(a))
    assert ids.shape == (3, 32)
    assert ids.min() >= 0 and ids.max() < 8
    torch.testing.assert_close(channel.detach().sum(-1), torch.ones(3, 32))
    q0 = torch.tensor([[0.2, 1.0], [-0.5, -1.0], [0.4, 1.2]])
    predicted = model.performer(channel, q0)
    torch.testing.assert_close(predicted[:, 0], q0)
    assert torch.diff(predicted, dim=1).abs().max() <= 0.1 + 1e-6
    predicted.square().mean().backward()
    assert model.notator.head.weight.grad.abs().sum() > 0
    model.eval()
    discrete, ids = model.notator(features(a))
    torch.testing.assert_close(discrete, torch.nn.functional.one_hot(ids, 8).float())


def test_learning_pipeline_smoke(tmp_path):
    report = train(
        tmp_path / "train",
        epochs=1,
        train_count=6,
        val_count=6,
        test_count=6,
        batch_size=6,
        threads=2,
    )
    assert report["summary"]["test"]["count"] == 6
    checkpoint = torch.load(tmp_path / "train/best.pt", weights_only=True)
    assert checkpoint["config"]["tokens"] == 32
    assert checkpoint["config"]["symbols"] == 8


def test_variable_eos_truncation_and_integer_roundtrip():
    config = ModelConfig(steps=8, variable=True)
    model = MotionModel(config)
    a = torch.rand(2, 8, 2)
    with torch.no_grad():
        model.notator.head.weight.zero_()
        model.notator.head.bias.zero_()
        model.notator.head.bias[8] = 5
    model.train()
    channel, ids = model.notator(features(a))
    assert ids.shape == (2, 33)
    assert torch.all(ids[:, 0] < 8)
    assert torch.all(ids[:, 1] == 8)
    assert torch.all(ids[:, 2:] == -1)
    torch.testing.assert_close(channel[:, 1:].detach(), torch.zeros(2, 31, 8))
    torch.testing.assert_close(channel.detach(), model.channel_from_scores(ids))
    model.notator.mean_length.backward()
    assert model.notator.head.bias.grad[8].abs() > 0
    with torch.no_grad():
        model.notator.head.bias[8] = -100
    _, full = model.notator(features(a))
    assert torch.all(full[:, :32] < 8)
    assert torch.all(full[:, 32] == 8)


def test_invalid_eos_transport_is_rejected():
    import pytest

    model = MotionModel(ModelConfig(steps=8, variable=True))
    score = torch.full((1, 33), -1, dtype=torch.long)
    score[0, :2] = torch.tensor([2, 8])
    assert model.channel_from_scores(score).sum() == 1
    score[0, 3] = 5
    with pytest.raises(ValueError, match="suffix"):
        model.channel_from_scores(score)
    score[0, :] = 1
    with pytest.raises(ValueError, match="exactly one EOS"):
        model.channel_from_scores(score)


def test_eos_training_and_reload(tmp_path):
    from roboscore.evaluate import run

    report = train(
        tmp_path / "eos",
        epochs=1,
        train_count=6,
        val_count=6,
        test_count=6,
        batch_size=6,
        threads=2,
        variable=True,
    )
    reevaluated = run(tmp_path / "eos")
    assert report["summary"]["test"] == reevaluated
    assert 1 <= reevaluated["mean_length"] <= 32
