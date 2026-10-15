from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)

    def forward(self, queries, memory):
        queries = self.query_norm(queries.float())
        memory = self.memory_norm(memory.float())
        return self.attention(queries, memory, memory, need_weights=False)[0]


class MemoryResampler(nn.Module):
    def __init__(self, width, slots, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.attention = CrossAttention(width, heads)
        self.norm = nn.LayerNorm(width)

    def forward(self, token_states):
        queries = self.queries.unsqueeze(0).expand(token_states.shape[0], -1, -1)
        return self.norm(queries + self.attention(queries, token_states))


class MLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs))

    def forward(self, inputs):
        return self.layers(inputs.float())


class LOITModel(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone, self.config = backbone, config
        core = backbone.model
        if not hasattr(core, "layers"):
            raise ValueError("The backbone requires decoder layers at model.layers")
        width, depth = backbone.config.hidden_size, len(core.layers)
        if width % config.attention_heads:
            raise ValueError("The hidden width requires divisibility by the cross-attention head count")
        self.width = width
        self.memory = MemoryResampler(width, config.memory_slots, config.attention_heads)
        self.codebook = nn.Parameter(torch.randn(config.code_count, config.code_slots, width) * 0.02)
        self.association_attention = nn.ModuleDict({str(index): CrossAttention(width, config.attention_heads) for index in config.layers(depth, "association")})
        self.return_attention = nn.ModuleDict({str(index): CrossAttention(width, config.attention_heads) for index in config.layers(depth, "return")})
        strength = min(1 - 1e-6, max(1e-6, config.return_strength))
        self.return_gates = nn.ParameterDict({index: nn.Parameter(torch.tensor([math.log(strength / (1 - strength))])) for index in self.return_attention})
        self.transition = nn.Linear(width, config.code_count)
        features, options = width + 3, len(config.options)
        self.actor = MLP(features, config.controller_hidden, options)
        self.critic = MLP(features, config.controller_hidden, 1)
        self.baseline = MLP(features, config.controller_hidden, 1)
        self.utility = MLP(features, config.controller_hidden, 2 * options)
        initial = []
        for category in config.categories:
            probabilities = torch.tensor(config.initial_distributions.get(category, [1 / config.code_count] * config.code_count), dtype=torch.float32)
            if probabilities.shape != (config.code_count,) or not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.isclose(probabilities.sum(), torch.tensor(1.0)):
                raise ValueError("Each category requires a probability distribution over codes")
            initial.append(probabilities)
        self.register_buffer("initial", torch.stack(initial))
        self._conditioning = None
        self._handles = [layer.register_forward_hook(self._layer_hook(str(index))) for index, layer in enumerate(core.layers)]
        # Hook conditioning is scoped to a forward pass; checkpoint replays require an explicit captured context.
        if hasattr(backbone, "gradient_checkpointing_disable"):
            backbone.gradient_checkpointing_disable()

    @property
    def device(self):
        return self.codebook.device

    def _layer_hook(self, index):
        def hook(module, inputs, output):
            condition = self._conditioning
            if condition is None:
                return output
            branch, memory, code, coupling, return_override = condition
            hidden = output[0] if isinstance(output, tuple) else output
            if branch == "association" and index in self.association_attention and coupling != 0:
                hidden = hidden + coupling * self.association_attention[index](hidden, memory).to(hidden.dtype)
            elif branch == "return" and index in self.return_attention:
                strength = self.return_gates[index].sigmoid() if return_override is None else return_override
                hidden = hidden + strength * self.return_attention[index](hidden, self.codebook[code].unsqueeze(0)).to(hidden.dtype)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return hook

    @contextmanager
    def condition(self, branch, memory=None, code=None, coupling=0, return_override=None):
        if self._conditioning is not None:
            raise RuntimeError("Conditioned backbone calls require sequential execution")
        if branch == "association" and memory is None:
            raise ValueError("Association calls require residual memory")
        if branch in ["association", "return"] and (code is None or not 0 <= code < self.config.code_count):
            raise ValueError("A conditioned branch requires a valid code index")
        self._conditioning = (branch, memory, code, coupling, return_override)
        try:
            yield
        finally:
            self._conditioning = None

    def forward(self, input_ids, branch="focused", memory=None, code=None, coupling=0, past_key_values=None, use_cache=False, return_override=None):
        with self.condition(branch, memory, code, coupling, return_override):
            values = {"past_key_values": past_key_values, "use_cache": use_cache, "output_hidden_states": True, "return_dict": True}
            if branch == "association" and past_key_values is None:
                tokens = self.backbone.get_input_embeddings()(input_ids)
                latent = self.codebook[code].unsqueeze(0).expand(tokens.shape[0], -1, -1).to(tokens.dtype)
                values["inputs_embeds"] = torch.cat([latent, tokens], dim=1)
            else:
                values["input_ids"] = input_ids
            return self.backbone(**values)

    def encode_memory(self, input_ids):
        states = self(input_ids).hidden_states[-1]
        return self.memory(states)

    def controller_state(self, memory, features):
        values = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, 3)
        return torch.cat([values, memory.float().mean(dim=1)], dim=-1)

    def predicted_utility(self, summaries):
        gain, raw_cost = self.utility(summaries).chunk(2, dim=-1)
        cost = F.softplus(raw_cost)
        # Focused continuation is the paired reference and has zero incremental utility.
        zeros = torch.zeros_like(gain[..., :1])
        gain, cost = torch.cat([zeros, gain[..., 1:]], -1), torch.cat([zeros, cost[..., 1:]], -1)
        return gain, cost, gain - self.config.cost_weight * cost

    def select_trainable(self, stage):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        groups = {
            "focused": [self.backbone],
            "initialize": [self.memory, self.association_attention, self.return_attention, self.return_gates, self.transition],
            "association": [self.association_attention, self.transition, self.baseline],
            "return": [self.return_attention, self.return_gates],
            "utility": [self.utility],
            "controller": [self.actor, self.critic],
        }
        if stage not in groups:
            raise ValueError("Unknown training stage")
        for module in groups[stage]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        # Eq. (9) updates association attention and transitions with the learned representations held constant.
        self.codebook.requires_grad_(stage == "initialize")
        self.backbone.eval()
