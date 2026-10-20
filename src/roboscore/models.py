"""Discrete trajectory Notator and closed-loop joint-increment Performer."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def fk(q):
    a, b = q[..., 0], q.sum(-1)
    return torch.stack((a.cos() + b.cos(), a.sin() + b.sin()), -1)


def features(q, dt=0.05):
    p = fk(q)
    velocity = torch.cat((torch.zeros_like(q[:, :1]), q[:, 1:] - q[:, :-1]), 1) / dt
    return torch.cat((q / torch.pi, velocity / 2, (p - p[:, :1]) / 2), -1)


@dataclass
class ModelConfig:
    steps: int = 96
    tokens: int = 32
    symbols: int = 8
    hidden: int = 64
    dt: float = 0.05
    variable: bool = False

    def __post_init__(self):
        if self.steps < 3 or self.tokens < 1 or self.symbols < 2:
            raise ValueError("invalid sequence or vocabulary size")
        if self.hidden < 4 or self.hidden % 4 or self.dt <= 0:
            raise ValueError("invalid hidden size or time step")


class Notator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.hidden
        self.input = nn.Linear(6, h)
        self.position = nn.Parameter(torch.randn(config.steps, h) * 0.02)
        layer = nn.TransformerEncoderLayer(h, 4, 4 * h, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(config.tokens, h) * 0.02)
        self.readout = nn.MultiheadAttention(h, 4, batch_first=True)
        self.head = nn.Linear(h, config.symbols + int(config.variable))

    def forward(self, trajectory, temperature=1.0):
        memory = self.encoder(self.input(trajectory) + self.position)
        queries = self.queries.unsqueeze(0).expand(len(trajectory), -1, -1)
        slots, _ = self.readout(queries, memory, memory, need_weights=False)
        logits = self.head(slots + self.queries)
        if self.config.variable:
            # Require at least one motion symbol; EOS has ID `symbols`.
            logits = logits.clone()
            logits[:, 0, self.config.symbols] = -torch.inf
        soft = (logits / temperature).softmax(-1)
        ids = logits.argmax(-1)
        hard = F.one_hot(ids, logits.shape[-1]).to(soft.dtype)
        # Exact discrete forward pass, soft gradient in the backward pass.
        channel = hard + soft - soft.detach() if self.training else hard
        if self.config.variable:
            eos = channel[..., self.config.symbols]
            alive = torch.cumprod(torch.cat((torch.ones_like(eos[:, :1]), 1 - eos), 1), 1)[:, :-1]
            channel = channel[..., : self.config.symbols] * alive.unsqueeze(-1)
            # Append a guaranteed EOS at the capacity limit, pad everything after first EOS.
            eos_tail = torch.full_like(ids[:, :1], self.config.symbols)
            raw = torch.cat((ids, eos_tail), 1)
            stop = (raw == self.config.symbols).to(torch.int64).argmax(1)
            positions = torch.arange(self.config.tokens + 1, device=ids.device)
            ids = torch.where(positions[None] <= stop[:, None], raw, -1)
        self.mean_length = channel.sum(-1).sum(-1).mean()
        return channel, ids


class Performer(nn.Module):
    """Only the message and receiver's current state enter this policy."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.hidden
        self.embedding = nn.Parameter(torch.randn(config.symbols, 8) * 0.1)
        self.message = nn.Sequential(nn.Linear(config.tokens * 8, h), nn.Tanh())
        self.initial = nn.Linear(h + 4, h)
        self.cell = nn.GRUCell(h + 6, h)
        self.action = nn.Linear(h, 2)
        nn.init.normal_(self.action.weight, std=0.001)
        nn.init.zeros_(self.action.bias)

    def forward(self, channel, q0):
        context = self.message((channel @ self.embedding).flatten(1))
        return self.rollout(context, q0)

    def rollout(self, context, q0):
        """Integrate the policy with a fixed conditioning context and live joint state."""
        q = q0
        first_context = context if context.ndim == 2 else context[:, 0]
        state = torch.tanh(self.initial(torch.cat((first_context, q.sin(), q.cos()), -1)))
        states = [q]
        for t in range(self.config.steps - 1):
            current_context = context if context.ndim == 2 else context[:, t]
            progress = q.new_full((len(q), 1), t / (self.config.steps - 1))
            inputs = torch.cat((current_context, q.sin(), q.cos(), progress, progress**2), -1)
            state = self.cell(inputs, state)
            delta = torch.tanh(self.action(state)) * (2 * self.config.dt)
            q = q + delta
            states.append(q)
        return torch.stack(states, 1)


class MotionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.notator = Notator(config)
        self.performer = Performer(config)

    def channel_from_scores(self, scores):
        """Decode integer transport only. EOS and padding cannot carry hidden suffix data."""
        expected = self.config.tokens + int(self.config.variable)
        if scores.ndim != 2 or scores.shape[1] != expected or scores.dtype != torch.long:
            raise ValueError("invalid integer score shape or dtype")
        if not self.config.variable:
            if ((scores < 0) | (scores >= self.config.symbols)).any():
                raise ValueError("invalid motion symbol")
            return F.one_hot(scores, self.config.symbols).float()
        eos = scores == self.config.symbols
        if not (eos.sum(1) == 1).all():
            raise ValueError("each variable score requires exactly one EOS")
        stop = eos.long().argmax(1)
        if (stop < 1).any():
            raise ValueError("at least one motion symbol required")
        pos = torch.arange(expected, device=scores.device)[None]
        before, after = pos < stop[:, None], pos > stop[:, None]
        if (((scores < 0) | (scores >= self.config.symbols)) & before).any():
            raise ValueError("invalid payload")
        if ((scores != -1) & after).any():
            raise ValueError("non-padding suffix after EOS")
        payload = scores[:, : self.config.tokens]
        return (
            F.one_hot(payload.clamp(0, self.config.symbols - 1), self.config.symbols).float()
            * before[:, : self.config.tokens, None]
        )

    def decode(self, scores, q0):
        return self.performer(self.channel_from_scores(scores), q0)

    def forward(self, a, q0, temperature=1.0):
        channel, ids = self.notator(features(a, self.config.dt), temperature)
        return self.performer(channel, q0), ids


class DiameterPerformer(Performer):
    """Diagnostic policy: an exact continuous diameter replaces the discrete message."""

    def __init__(self, config):
        super().__init__(config)
        del self.embedding
        self.message = nn.Sequential(nn.Linear(1, config.hidden), nn.Tanh())

    def forward(self, diameter, q0):
        if diameter.ndim != 1 or len(diameter) != len(q0):
            raise ValueError("expected one diameter per initial state")
        return self.rollout(self.message(diameter[:, None]), q0)
