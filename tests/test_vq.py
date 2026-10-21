import numpy as np
import pytest
import torch

from roboscore.models import ModelConfig
from roboscore.train import dataset
from roboscore.vq import VQConfig, VQModel, reevaluate, train


def test_vq_integer_transport_causality_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    model = VQModel(VQConfig())
    motion = torch.randn(2, 95, 2) * 0.015
    model.initialize_codebook(motion)
    q0 = torch.tensor([[0.2, 1.1], [-0.4, -1.2]])
    q, ids, reconstruction, vq = model(motion, q0)
    torch.testing.assert_close(q, model.decode(ids, q0), atol=2e-6, rtol=1e-5)
    assert ids.shape == (2, 95) and ids.min() >= 0 and ids.max() < 8
    assert torch.diff(q, dim=1).abs().max() <= 0.1 + 1e-6
    changed = motion.clone()
    changed[:, 50:] *= -1
    torch.testing.assert_close(model.encode(changed)[:, :50], ids[:, :50])
    altered_ids = ids.clone()
    altered_ids[:, 50:] = (altered_ids[:, 50:] + 1) % 8
    torch.testing.assert_close(
        model.decode(altered_ids, q0)[:, :51], q[:, :51], atol=2e-6, rtol=1e-5
    )
    (q.square().mean() + reconstruction + vq).backward()
    for component in (model.translator, model.performer, model.reconstruct):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    assert model.performer.embedding.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model.decode(torch.full((2, 95), 8, dtype=torch.long), q0)


def test_vq_training_reload_and_hash_refusal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i, name in enumerate(("train", "val", "test")):
        dataset(6, 30 + i, ModelConfig(tokens=1), "circles", source / f"{name}_demonstrations.npz")
    report = train(source, tmp_path / "run", epochs=1, pretrain_epochs=1, min_epochs=1, threads=2)
    assert reevaluate(tmp_path / "run", source) == report["summary"]["test"]
    assert sum(report["code_usage"]["counts"]) == 6 * 95
    path = source / "test_demonstrations.npz"
    with np.load(path) as data:
        values = {k: data[k].copy() for k in data.files}
    values["q_a"][0, 0, 0] += 0.001
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="hashes"):
        reevaluate(tmp_path / "run", source)


def test_matched_global_and_local_clock_models():
    from roboscore.aligned import weight_hash

    torch.set_num_threads(2)
    torch.manual_seed(123)
    local = VQModel(VQConfig(mode="local", auxiliary_clock=True))
    torch.manual_seed(123)
    global_model = VQModel(VQConfig(mode="global", auxiliary_clock=True))
    assert weight_hash(local) == weight_hash(global_model)
    assert sum(p.numel() for p in local.parameters()) == sum(
        p.numel() for p in global_model.parameters()
    )
    motion = torch.randn(4, 95, 2) * 0.015
    q0 = torch.tensor([[0.2, 1.1]]).expand(4, -1)
    for model, tokens in ((local, 95), (global_model, 1)):
        model.initialize_codebook(motion)
        q, ids, reconstruction, vq = model(motion, q0)
        assert ids.shape == (4, tokens)
        torch.testing.assert_close(q, model.decode(ids, q0), atol=2e-6, rtol=1e-5)
        (q.square().mean() + reconstruction + vq).backward()
        assert model.translator.output.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="time-conditioned"):
        VQConfig(mode="global")


def test_global_clock_training_and_identity_control(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i, name in enumerate(("train", "val", "test")):
        dataset(8, 60 + i, ModelConfig(tokens=1), "circles", source / f"{name}_demonstrations.npz")
    report = train(
        source,
        tmp_path / "run",
        epochs=1,
        pretrain_epochs=1,
        min_epochs=1,
        threads=2,
        mode="global",
        auxiliary_clock=True,
    )
    assert report["summary"]["test"] == report["summary"]["permute_steps"]
    assert reevaluate(tmp_path / "run", source) == report["summary"]["test"]
