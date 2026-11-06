from pathlib import Path
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
import uuid
import torch

from .config import Config
from .models import LOITModel
from .language import LanguageModel


def load_language(config, checkpoint=None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    config = replace(config)
    set_seed(config.seed)
    metadata = {}
    if checkpoint is not None:
        root = Path(checkpoint)
        saved = Config.load(root / "loit.yaml")
        architecture = ["backbone", "memory_slots", "code_count", "code_slots", "attention_heads",
                        "association_layers", "return_layers", "layer_fractions", "categories", "durations",
                        "couplings", "controller_hidden"]
        for name in architecture:
            if getattr(config, name) != getattr(saved, name):
                raise ValueError(f"The requested {name} differs from the checkpoint")
        if config.revision is not None and saved.revision is not None and config.revision != saved.revision:
            raise ValueError("The requested backbone revision differs from the checkpoint")
        config.revision = saved.revision if config.revision is None else config.revision
        for name in ["gate_threshold", "utility_margin"]:
            if getattr(config, name) is None:
                setattr(config, name, getattr(saved, name))
        config.__post_init__()
        backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(root / "backbone"), dtype=torch.float32)
        if (root / "backbone" / "generation_config.json").is_file():
            backbone.generation_config = GenerationConfig.from_pretrained(root / "backbone")
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        if (root / "training.json").is_file():
            metadata = json.loads((root / "training.json").read_text(encoding="utf-8"))
    else:
        backbone = AutoModelForCausalLM.from_pretrained(config.backbone, revision=config.revision, dtype=torch.float32)
        config.revision = getattr(backbone.config, "_commit_hash", None) or config.revision
        tokenizer = AutoTokenizer.from_pretrained(config.backbone, revision=config.revision)
    model = LOITModel(backbone, config)
    if checkpoint is not None:
        weights = torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    model.to(config.device)
    model.eval()
    language = LanguageModel(model, tokenizer)
    language.checkpoint_id = metadata.get("checkpoint_id")
    if checkpoint is not None and language.checkpoint_id is None:
        language.checkpoint_id = file_digest(Path(checkpoint) / "model.pt")
    return language


def save_checkpoint(language, output, state=None, metadata=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    language.config.save(root / "loit.yaml")
    language.model.backbone.config.save_pretrained(root / "backbone")
    language.model.backbone.generation_config.save_pretrained(root / "backbone")
    language.tokenizer.save_pretrained(root / "tokenizer")
    torch.save(language.model.state_dict() if state is None else state, root / "model.pt.tmp")
    os.replace(root / "model.pt.tmp", root / "model.pt")
    record = {**(metadata or {}), "checkpoint_id": uuid.uuid4().hex,
              "source_checkpoint_id": getattr(language, "checkpoint_id", None),
              "library_versions": software_versions()}
    write_json(root / "training.json", record)
    language.checkpoint_id = record["checkpoint_id"]


def apply_calibration(config, path, checkpoint_id=None):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint_id is not None and values.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Calibration requires the checkpoint used to fit its thresholds")
    if checkpoint_id is not None and values.get("configuration_digest") != calibration_digest(config):
        raise ValueError("Calibration requires the same proposal, gate, and utility configuration")
    config.gate_threshold = float(values["gate_threshold"])
    config.utility_margin = float(values["utility_margin"])
    config.__post_init__()
    return config


def calibration_digest(config):
    values = asdict(config)
    for name in ["device", "seed", "gate_threshold", "utility_margin", "learning_rate", "backbone_learning_rate",
                 "weight_decay", "adam_betas", "gradient_clip", "warmup_fraction", "minimum_lr_ratio", "epochs",
                 "batch_size", "global_batch_size", "ppo_clip", "entropy_weight", "cost_regression_weight"]:
        values.pop(name)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions():
    versions = {}
    for name in ["torch", "transformers", "accelerate", "numpy", "lm-format-enforcer"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
