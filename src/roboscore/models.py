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
        self.head = nn.Linear(h, config.symbols)

    def forward(self, trajectory, temperature=1.0):
        memory = self.encoder(self.input(trajectory) + self.position)
        queries = self.queries.unsqueeze(0).expand(len(trajectory), -1, -1)
        slots, _ = self.readout(queries, memory, memory, need_weights=False)
        logits = self.head(slots + self.queries)
        soft = (logits / temperature).softmax(-1)
        ids = logits.argmax(-1)
        hard = F.one_hot(ids, self.config.symbols).to(soft.dtype)
        # Exact discrete forward pass, soft gradient in the backward pass.
        channel = hard + soft - soft.detach() if self.training else hard
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
        q = q0
        state = torch.tanh(self.initial(torch.cat((context, q.sin(), q.cos()), -1)))
        states = [q]
        for t in range(self.config.steps - 1):
            progress = q.new_full((len(q), 1), t / (self.config.steps - 1))
            inputs = torch.cat((context, q.sin(), q.cos(), progress, progress**2), -1)
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

    def forward(self, a, q0, temperature=1.0):
        channel, ids = self.notator(features(a, self.config.dt), temperature)
        return self.performer(channel, q0), ids
