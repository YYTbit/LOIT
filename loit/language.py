from dataclasses import dataclass
from copy import deepcopy

import torch
from torch.nn import functional as F

from . import prompts


class ComputationBudgetExceeded(RuntimeError):
    pass


@dataclass
class FocusedCache:
    values: object
    logits: torch.Tensor
    prefix_ids: list[int]


@dataclass
class Completion:
    text: str
    prompt_ids: list[int]
    token_ids: list[int]
    log_probability: float
    processed_tokens: int


class LanguageModel:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.config = model, tokenizer, model.config
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def ids(self, text, reserve=0, chat=True):
        if chat and self.tokenizer.chat_template:
            values = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        else:
            values = self.tokenizer.encode(text, add_special_tokens=chat)
        limit = self.config.context_tokens - reserve
        if limit < 1 or not values:
            raise ValueError("The context budget requires a nonempty prompt and reserved response space")
        if len(values) > limit:
            raise ValueError("The prompt exceeds the context budget; shorten the recorded history while retaining the question")
        return values

    def tensor(self, values):
        return torch.tensor([values], dtype=torch.long, device=self.model.device)

    def memory(self, state):
        values = self.ids(state.canonical(), reserve=self.config.code_slots)
        return self.model.encode_memory(self.tensor(values)), len(values)

    @torch.no_grad()
    def prefill(self, state):
        self.model.backbone.eval()
        values = self.ids(prompts.focused(state), reserve=self.config.proposal_tokens + 128)
        output = self.model(self.tensor(values), use_cache=True)
        return FocusedCache(output.past_key_values, output.logits[0, -1].detach(), values)

    def grammar(self, schema):
        if schema is None or not self.config.grammar:
            return None
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return build_transformers_prefix_allowed_tokens_fn(self.tokenizer, JsonSchemaParser(schema))

    @torch.no_grad()
    def complete(self, prompt, max_tokens, seed=0, branch="focused", memory=None, code=None, coupling=0, cache=None, schema=None, position_budget=None):
        self.model.backbone.eval()
        if max_tokens < 1:
            raise ValueError("Generation requires a positive response budget")
        latent = self.config.code_slots if branch == "association" else 0
        prefix = self.ids(prompt, reserve=max_tokens + latent, chat=cache is None) if prompt else []
        if position_budget is not None:
            if position_budget < 1 or len(prefix) + latent > position_budget:
                raise ComputationBudgetExceeded("The remaining decoder-position budget cannot encode this prompt")
            max_tokens = min(max_tokens, position_budget - len(prefix) - latent + 1)
        if cache is not None and len(cache.prefix_ids) + len(prefix) + max_tokens > self.config.context_tokens:
            raise ValueError("The saved focused cache and return response exceed the context budget")
        values = deepcopy(cache.values) if cache is not None else None
        if prefix:
            output = self.model(self.tensor(prefix), branch, memory, code, coupling, values, use_cache=True)
            values, logits = output.past_key_values, output.logits[0, -1]
        elif cache is not None and branch == "focused":
            logits = cache.logits
        else:
            raise ValueError("Generation requires a prompt or a focused cache")
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        grammar = self.grammar(schema)
        response, probability = [], 0.0
        processed = len(prefix) + latent
        eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        for step in range(max_tokens):
            logits = logits.float() / self.config.temperature
            if grammar is not None:
                allowed = grammar(0, self.tensor(prefix + response)[0])
                masked = torch.full_like(logits, -torch.inf)
                masked[allowed] = logits[allowed]
                logits = masked
            if not torch.isfinite(logits).any():
                raise RuntimeError("Generation has no admissible vocabulary token")
            log_probs = logits.log_softmax(-1)
            token = int(torch.multinomial(log_probs.exp(), 1, generator=generator))
            response.append(token)
            probability += float(log_probs[token])
            if token in eos_ids or step == max_tokens - 1:
                break
            output = self.model(self.tensor([token]), branch, memory, code, coupling, values, use_cache=True)
            values, logits = output.past_key_values, output.logits[0, -1]
            processed += 1
        return Completion(self.tokenizer.decode(response, skip_special_tokens=True), prefix, response, probability, processed)

    def continuation_loss(self, prompt, response, branch="focused", memory=None, code=None, coupling=0):
        target = self.tokenizer.encode(response, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            target.append(eos)
        if not target:
            raise ValueError("Supervised continuations require response tokens")
        latent = self.config.code_slots if branch == "association" else 0
        prefix = self.ids(prompt, reserve=len(target) + latent)
        output = self.model(self.tensor(prefix + target), branch, memory, code, coupling)
        start = len(prefix) + latent - 1
        logits = output.logits[:, start:start + len(target)].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.tensor(target).reshape(-1))

    def return_loss(self, state, code, response):
        # The scientific prefix is encoded without return attention, matching the saved focused cache.
        cache = self.prefill(state)
        target = self.tokenizer.encode(response, add_special_tokens=False)
        if isinstance(self.tokenizer.eos_token_id, int):
            target.append(self.tokenizer.eos_token_id)
        prefix = self.ids(prompts.proposal(), reserve=len(target), chat=False)
        if len(cache.prefix_ids) + len(prefix) + len(target) > self.config.context_tokens:
            raise ValueError("The supervised return exceeds the context budget")
        output = self.model(self.tensor(prefix + target), "return", code=code, past_key_values=deepcopy(cache.values))
        logits = output.logits[:, len(prefix) - 1:len(prefix) - 1 + len(target)].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.tensor(target).reshape(-1))

    def score_activity(self, memory, previous_code, prompt_ids, response_ids, coupling):
        if not response_ids or not prompt_ids or len(prompt_ids) + len(response_ids) + self.config.code_slots > self.config.context_tokens:
            raise ValueError("The recorded activity continuation is empty or exceeds the context budget")
        output = self.model(self.tensor(prompt_ids + response_ids), "association", memory, previous_code, coupling)
        start = self.config.code_slots + len(prompt_ids)
        logits = output.logits[0, start - 1:start - 1 + len(response_ids)].float() / self.config.temperature
        selected = logits.log_softmax(-1).gather(-1, self.tensor(response_ids)[0, :, None]).sum()
        # Pool response positions; prompt and latent-prefix states are excluded.
        pooled = output.hidden_states[-1][:, start:start + len(response_ids)].float().mean(dim=1)
        return selected, self.model.transition(pooled)[0]
