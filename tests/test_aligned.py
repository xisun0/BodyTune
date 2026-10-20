import numpy as np
import torch

from roboscore.aligned import AlignedConfig, AlignedModel, reevaluate, train, weight_hash
from roboscore.data import generate
from roboscore.kinematics import Arm
from roboscore.models import ModelConfig
from roboscore.train import dataset


def test_modes_have_identical_parameter_shapes_and_initial_values():
    torch.manual_seed(123)
    a = AlignedModel(AlignedConfig(mode="global"))
    torch.manual_seed(123)
    b = AlignedModel(AlignedConfig(mode="local"))
    assert weight_hash(a) == weight_hash(b)
    assert a.config.tokens == 1 and b.config.tokens == 95


def test_local_encoder_and_performer_cannot_read_future_tokens():
    torch.set_num_threads(2)
    model = AlignedModel(AlignedConfig(mode="local"))
    model.eval()
    motion = torch.rand(2, 95, 2) * 0.02
    changed = motion.clone()
    changed[:, 50:] *= -1
    with torch.no_grad():
        _, ids = model.notator(motion)
        _, changed_ids = model.notator(changed)
        torch.testing.assert_close(ids[:, :50], changed_ids[:, :50])
        scores = torch.randint(0, 8, (2, 95))
        altered = scores.clone()
        altered[:, 50:] = (altered[:, 50:] + 1) % 8
        q0 = torch.tensor([[0.2, 1.1], [-0.5, -1.3]])
        q = model.decode(scores, q0)
        other = model.decode(altered, q0)
        torch.testing.assert_close(q[:, :51], other[:, :51], atol=0, rtol=0)
        assert torch.diff(q, dim=1).abs().max() <= 0.1 + 1e-6


def test_displacements_are_invariant_to_joint_realization():
    pairs, _ = generate(6, task="circles")
    arm = Arm()
    for pair in pairs:
        np.testing.assert_allclose(
            np.diff(arm.fk(pair.q_a), axis=0), np.diff(arm.fk(pair.q_b), axis=0), atol=1e-12
        )


def test_both_training_modes_and_integer_reload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i, name in enumerate(("train", "val", "test")):
        dataset(6, 20 + i, ModelConfig(tokens=1), "circles", source / f"{name}_demonstrations.npz")
    reports = []
    for mode in ("global", "local", "binned"):
        report = train(
            source, tmp_path / mode, mode=mode, epochs=1, min_epochs=1, batch_size=6, threads=2
        )
        result = reevaluate(tmp_path / mode, source)
        assert result == report["summary"]["test"]
        reports.append(report)
    assert (
        reports[0]["training"]["initial_weights_sha256"]
        == reports[1]["training"]["initial_weights_sha256"]
    )
    assert reports[0]["data"]["sha256"] == reports[1]["data"]["sha256"]


def test_same_local_displacement_has_same_symbol_at_any_step():
    model = AlignedModel(AlignedConfig(mode="local"))
    model.eval()
    motion = torch.rand(1, 95, 2) * 0.02
    motion[:, 80] = motion[:, 3]
    with torch.no_grad():
        channel, ids = model.notator(motion)
    torch.testing.assert_close(channel[:, 80], channel[:, 3], atol=0, rtol=0)
    assert ids[0, 80] == ids[0, 3]


def test_fixed_diameter_bins_and_matching_performer_initialization():
    from roboscore.aligned import DiameterBins

    centers = torch.linspace(0.24, 0.70, 9)
    centers = (centers[:-1] + centers[1:]) / 2
    _, ids = DiameterBins()(centers)
    assert ids[:, 0].tolist() == list(range(8))
    _, endpoints = DiameterBins()(torch.tensor([0.24, 0.70]))
    assert endpoints[:, 0].tolist() == [0, 7]
    torch.manual_seed(123)
    a = AlignedModel(AlignedConfig(mode="global"))
    torch.manual_seed(123)
    b = AlignedModel(AlignedConfig(mode="binned"))
    assert weight_hash(a.performer) == weight_hash(b.performer)
