import string

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from loit.config import Config
from loit.language import LanguageModel
from loit.models import LOITModel


@pytest.fixture
def tiny():
    torch.set_num_threads(1)

    def build(family="qwen3", **overrides):
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        vocab = {token: index for index, token in enumerate(["[UNK]", "[EOS]", "[BOS]"] + list(dict.fromkeys(string.printable)))}
        raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        raw.decoder = decoders.Fuse()
        raw.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]", pad_token="[EOS]")
        types = {"qwen3": (Qwen3Config, Qwen3ForCausalLM), "llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM)}
        config_type, model_type = types[family]
        architecture = config_type(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=4096, bos_token_id=2, eos_token_id=1, pad_token_id=1,
                                   attention_dropout=0.0)
        settings = dict(backbone="tiny", precision="no", attention_heads=2, code_count=3, code_slots=2,
                        memory_slots=2, durations=[1, 2], couplings=[0, 1], categories=["brainstorming"],
                        context_tokens=4096, activity_tokens=4, proposal_tokens=256, action_tokens=4,
                        gate_tokens=96, grammar=False, controller_hidden=8, gate_threshold=0.5,
                        utility_margin=0.0, epochs=1, gradient_accumulation=1)
        settings.update(overrides)
        torch.manual_seed(7)
        model = LOITModel(model_type(architecture), Config(**settings))
        model.eval()
        return LanguageModel(model, tokenizer)

    return build
