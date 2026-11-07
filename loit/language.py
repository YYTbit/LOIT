from dataclasses import dataclass
from copy import deepcopy

import torch
from torch.nn import functional as F

from . import prompts
from .decoding import SamplingPolicy, StructuredOutput, association_scores, final_answer, native_reasoning_prefix, supervised_response


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
    finish_reason: str = "length"
    reasoning_complete: bool = True


class LanguageModel:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.config = model, tokenizer, model.config
        self.reasoning_end = tokenizer.get_vocab().get("</think>")
        self._grammar_tokens = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def ids(self, text, reserve=0, chat=True):
        if chat and self.tokenizer.chat_template:
            values = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
        else:
            values = self.tokenizer.encode(text, add_special_tokens=chat)
        limit = self.config.context_tokens - reserve
        if not values:
            raise ValueError("The context budget requires a nonempty prompt")
        if limit < 1:
            raise ValueError("The context budget cannot reserve this continuation; adjust generation limits while retaining the question")
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
        values = self.focused_ids(state)
        output = self.model(self.tensor(values), use_cache=True)
        return FocusedCache(output.past_key_values, output.logits[0, -1].detach(), values)

    def focused_ids(self, state):
        suffix = len(self.ids(prompts.proposal(), chat=False))
        reserve = max(self.config.action_tokens, suffix + self.config.proposal_tokens)
        return self.ids(prompts.focused(state), reserve=reserve)

    def grammar(self, schema):
        if schema is None or not self.config.grammar:
            return None
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        if self._grammar_tokens is None:
            self._grammar_tokens = build_token_enforcer_tokenizer_data(self.tokenizer)
        return StructuredOutput(self.tokenizer, self._grammar_tokens, schema)

    @torch.no_grad()
    def complete(self, prompt, max_tokens, seed=0, branch="focused", memory=None, code=None, coupling=0, cache=None, schema=None, position_budget=None):
        self.model.backbone.eval()
        if max_tokens < 1:
            raise ValueError("Generation requires a positive response budget")
        if branch == "association" and schema is not None and self.config.grammar:
            raise ValueError("Association sampling requires the same full-support response policy used for replay")
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
        history = (cache.prefix_ids if cache is not None else []) + prefix
        policy = SamplingPolicy(self.model.backbone, self.config, branch, len(history), max_tokens,
                                grammar, self.model.device, self.tokenizer)
        response, probability = [], 0.0
        processed = len(prefix) + latent
        finish_reason = "length"
        for step in range(max_tokens):
            scores = policy.scores(self.tensor(history + response), logits)
            token, log_probability = policy.draw(scores, generator)
            response.append(token)
            probability += log_probability
            next_token = self.tensor([token])
            if policy.stopping(self.tensor(history + response), None).item():
                finish_reason = "length" if step == max_tokens - 1 and token not in policy.eos else "stop"
                break
            if step == max_tokens - 1:
                break
            output = self.model(next_token, branch, memory, code, coupling, values, use_cache=True)
            values, logits = output.past_key_values, output.logits[0, -1]
            processed += 1
        text, reasoning_complete = final_answer(self.tokenizer, response, self.reasoning_end,
                                                native_reasoning_prefix(self.tokenizer))
        return Completion(text, prefix, response, probability, processed, finish_reason, reasoning_complete)

    def continuation_loss(self, prompt, response, branch="focused", memory=None, code=None, coupling=0):
        target = self.tokenizer.encode(supervised_response(self.tokenizer, response), add_special_tokens=False)
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
        target = self.tokenizer.encode(supervised_response(self.tokenizer, response), add_special_tokens=False)
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
        logits = association_scores(output.logits[0, start - 1:start - 1 + len(response_ids)], self.config.temperature)
        selected = logits.log_softmax(-1).gather(-1, self.tensor(response_ids)[0, :, None]).sum()
        # Pool response positions; prompt and latent-prefix states are excluded.
        pooled = output.hidden_states[-1][:, start:start + len(response_ids)].float().mean(dim=1)
        return selected, self.model.transition(pooled)[0]
