from copy import deepcopy

import torch


def association_scores(scores, temperature):
    from transformers import TemperatureLogitsWarper
    return TemperatureLogitsWarper(float(temperature))(None, scores.float())


def native_reasoning_prefix(tokenizer):
    if not tokenizer.chat_template or not {"<think>", "</think>"}.issubset(tokenizer.get_vocab()):
        return False
    messages = [{"role": "user", "content": ""}]
    before = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    after = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    suffix = after[len(before):] if after.startswith(before) else ""
    return suffix.rfind("<think>") > suffix.rfind("</think>")


def supervised_response(tokenizer, text):
    if native_reasoning_prefix(tokenizer):
        if text.startswith("<think>"):
            return text[len("<think>"):]
        if not text.startswith("</think>"):
            return "</think>\n\n" + text
    return text


def reasoning_started(tokenizer, response, begin):
    if begin is None or begin not in response:
        return False
    prefix = response[:response.index(begin)]
    return not tokenizer.decode(prefix, skip_special_tokens=False).strip()


class StructuredOutput:
    def __init__(self, tokenizer, tokenizer_data, schema):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        self.direct = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))
        self.returned = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))
        vocabulary = tokenizer.get_vocab()
        self.begin = vocabulary.get("<think>")
        self.end = vocabulary.get("</think>")
        self.prefix_open = native_reasoning_prefix(tokenizer)
        self.prefix_length = None
        self.tokenizer = tokenizer

    def __call__(self, batch, tokens):
        if self.prefix_length is None:
            self.prefix_length = len(tokens)
        response = tokens[self.prefix_length:].tolist()
        if self.prefix_open or reasoning_started(self.tokenizer, response, self.begin):
            if self.end is not None and self.end in response:
                # The second native enforcer starts at the completed reasoning prefix.
                return self.returned(batch, tokens)
            return None
        allowed = self.direct(batch, tokens)
        if not self.tokenizer.decode(response, skip_special_tokens=False).strip() and self.begin is not None and self.end is not None:
            allowed = [*allowed, self.begin]
        return allowed


class SchemaLogitsProcessor:
    def __init__(self, grammar):
        from transformers import PrefixConstrainedLogitsProcessor

        self.grammar = grammar
        self.allowed = None
        self.native = PrefixConstrainedLogitsProcessor(self.allowed_tokens, num_beams=1)

    def allowed_tokens(self, batch, tokens):
        return self.allowed

    def __call__(self, input_ids, scores):
        self.allowed = self.grammar(0, input_ids[0])
        if self.allowed is None:
            return scores
        return self.native(input_ids, scores)


class SamplingPolicy:
    def __init__(self, backbone, config, branch, prefix_length, max_tokens, grammar, device, tokenizer=None):
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

        native = backbone.generation_config
        if branch == "association":
            # Full support keeps the sampled response and its replay likelihood identical in Eq. (9).
            settings = GenerationConfig(do_sample=True, temperature=float(config.temperature), top_k=0,
                                        eos_token_id=native.eos_token_id, bos_token_id=native.bos_token_id,
                                        pad_token_id=native.pad_token_id)
        else:
            settings = deepcopy(native)
            unused = settings.update(**config.generation)
            if unused:
                raise ValueError("Unknown native generation settings: " + ", ".join(sorted(unused)))
        if settings.num_beams != 1 or settings.num_return_sequences != 1 or settings.penalty_alpha:
            raise ValueError("Conditioned streams require single-sequence sampling or greedy decoding")
        settings.max_new_tokens = max_tokens
        settings.max_length = prefix_length + max_tokens
        settings.validate()
        backbone._prepare_special_tokens(settings, device=device)
        self.eos = set(settings._eos_token_tensor.tolist()) if settings._eos_token_tensor is not None else set()
        processors = LogitsProcessorList([SchemaLogitsProcessor(grammar)]) if grammar is not None else LogitsProcessorList()
        self.processors = backbone._get_logits_processor(
            settings, input_ids_seq_length=prefix_length, encoder_input_ids=None,
            prefix_allowed_tokens_fn=None, logits_processor=processors, device=device)
        self.stopping = backbone._get_stopping_criteria(
            settings, stopping_criteria=StoppingCriteriaList(), tokenizer=tokenizer)
        self.settings = settings

    def scores(self, history, logits):
        return self.processors(history, logits.float().unsqueeze(0))[0]

    def draw(self, scores, generator):
        if not torch.isfinite(scores).any():
            raise RuntimeError("Generation has no admissible vocabulary token")
        log_probabilities = scores.log_softmax(-1)
        if self.settings.do_sample:
            token = int(torch.multinomial(log_probabilities.exp(), 1, generator=generator))
            probability = float(log_probabilities[token])
        else:
            token, probability = int(scores.argmax()), 0.0
        return token, probability


def final_answer(tokenizer, response, reasoning_end, prefix_open=False):
    begin = tokenizer.get_vocab().get("<think>")
    if prefix_open or reasoning_started(tokenizer, response, begin):
        if reasoning_end is not None and reasoning_end in response:
            return tokenizer.decode(response[response.index(reasoning_end) + 1:], skip_special_tokens=True), True
        return "", False
    return tokenizer.decode(response, skip_special_tokens=True), True
