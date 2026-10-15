from pathlib import Path
import json
import torch

from .config import Config
from .models import LOITModel
from .language import LanguageModel


def load_language(config, checkpoint=None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if checkpoint is not None:
        root = Path(checkpoint)
        saved = Config.load(root / "loit.yaml")
        saved.device = config.device
        saved.gate_threshold = config.gate_threshold if config.gate_threshold is not None else saved.gate_threshold
        saved.utility_margin = config.utility_margin if config.utility_margin is not None else saved.utility_margin
        config = saved
        backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root / "backbone"))
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    else:
        backbone = AutoModelForCausalLM.from_pretrained(config.backbone, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = LOITModel(backbone, config)
    if checkpoint is not None:
        weights = torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    model.to(config.device)
    model.eval()
    return LanguageModel(model, tokenizer)


def save_checkpoint(language, output, state=None, metadata=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    language.config.save(root / "loit.yaml")
    language.model.backbone.config.save_pretrained(root / "backbone")
    language.tokenizer.save_pretrained(root / "tokenizer")
    torch.save(language.model.state_dict() if state is None else state, root / "model.pt")
    (root / "training.json").write_text(json.dumps(metadata or {}, indent=2), encoding="utf-8")


def apply_calibration(config, path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    config.gate_threshold = float(values["gate_threshold"])
    config.utility_margin = float(values["utility_margin"])
    config.__post_init__()
    return config
