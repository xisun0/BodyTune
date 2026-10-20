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
