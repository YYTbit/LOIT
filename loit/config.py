from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class Option:
    duration: int = 0
    coupling: float = 0.0
    category: str = ""


@dataclass
class Config:
    backbone: str = "Qwen/Qwen3-8B"
    device: str = "cpu"
    precision: str = "bf16"
    memory_slots: int = 4
    code_count: int = 8
    code_slots: int = 4
    attention_heads: int = 8
    association_layers: list[int] = field(default_factory=list)
    return_layers: list[int] = field(default_factory=list)
    layer_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    categories: list[str] = field(default_factory=lambda: ["brainstorming", "generation", "open_qa", "summarization"])
    durations: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 12])
    couplings: list[float] = field(default_factory=lambda: [0, 0.25, 0.5, 0.75, 1])
    topic_change: float = 0.24
    initial_distributions: dict[str, list[float]] = field(default_factory=dict)
    return_strength: float = 0.5
    proposal_count: int = 4
    context_tokens: int = 4096
    activity_tokens: int = 128
    proposal_tokens: int = 256
    action_tokens: int = 128
    gate_tokens: int = 96
    temperature: float = 1.0
    grammar: bool = True
    gate_threshold: float | None = None
    utility_margin: float | None = None
    gate_weights: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    controller_hidden: int = 256
    idea_weight: float = 1.0
    cost_weight: float = 1.0
    cost_scale: float = 1000.0
    cost_regression_weight: float = 1.0
    learning_rate: float = 3e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.03
    minimum_lr_ratio: float = 0.1
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 128
    ppo_clip: float = 0.2
    entropy_weight: float = 0.01
    seed: int = 0

    def __post_init__(self):
        for name in ["memory_slots", "code_count", "code_slots", "attention_heads", "proposal_count", "context_tokens", "activity_tokens", "proposal_tokens", "action_tokens", "gate_tokens", "batch_size", "gradient_accumulation", "epochs"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} requires a positive integer")
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("Activity categories require distinct names")
        if not self.durations or len(set(self.durations)) != len(self.durations) or any(type(k) is not int or k < 1 for k in self.durations):
            raise ValueError("Incubation durations require positive integers")
        if not self.couplings or any(not 0 <= a <= 1 for a in self.couplings):
            raise ValueError("Coupling values require the interval [0, 1]")
        if not 0 <= self.topic_change <= 1 or not 0 <= self.return_strength <= 1:
            raise ValueError("Topic-change and return strengths require [0, 1]")
        if any(not math.isfinite(value) for value in [self.temperature, self.cost_scale, self.cost_weight, self.idea_weight]) or self.temperature <= 0 or self.cost_scale <= 0 or self.cost_weight < 0 or self.idea_weight < 0:
            raise ValueError("Temperature, cost scale, and reward weights are invalid")
        if len(self.gate_weights) != 4 or min(self.gate_weights) < 0 or not math.isclose(sum(self.gate_weights), 1):
            raise ValueError("Four nonnegative gate weights require unit sum")
        if self.gate_threshold is not None and not 0 <= self.gate_threshold <= 1:
            raise ValueError("The gate threshold requires [0, 1]")
        if self.utility_margin is not None and (not math.isfinite(self.utility_margin) or self.utility_margin < 0):
            raise ValueError("The deployment margin requires a nonnegative value")
        if not self.layer_fractions or any(not 0 < value <= 1 for value in self.layer_fractions):
            raise ValueError("Cross-attention depth fractions require (0, 1]")
        if not 0 < self.ppo_clip < 1 or self.entropy_weight < 0:
            raise ValueError("PPO clipping and entropy settings are invalid")

    @property
    def options(self):
        # Index zero preserves focused continuation and wins exact policy ties.
        return [Option()] + [Option(k, a, d) for k, a, d in product(self.durations, self.couplings, self.categories)]

    def layers(self, depth, branch):
        specified = self.association_layers if branch == "association" else self.return_layers
        indices = specified or sorted({max(0, min(depth - 1, int(depth * fraction) - 1)) for fraction in self.layer_fractions})
        if any(index < 0 or index >= depth for index in indices):
            raise ValueError("Cross-attention layer indices exceed the backbone depth")
        return indices

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
