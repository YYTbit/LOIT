from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import random
import re


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Record {number} requires an object")
                records.append(value)
    if not records:
        raise ValueError("The record file is empty")
    return records


def write_jsonl(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def stable_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class Proposal:
    hypothesis: str
    competing_explanation: str
    discriminating_test: str
    prediction: str
    competing_prediction: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("Scientific proposals require five nonempty text fields")

    def text(self):
        return stable_text(asdict(self))


@dataclass
class ScientificState:
    question: str
    observations: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    tools: object = field(default_factory=list)
    task_id: str = ""
    split: str = "unspecified"
    remaining_fraction: float = 1.0
    actions: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("A scientific state requires a question")
        if self.split not in ["train", "dev", "validation", "test", "unspecified"]:
            raise ValueError("The scientific-state split is invalid")
        if not math.isfinite(self.remaining_fraction) or not 0 <= self.remaining_fraction <= 1:
            raise ValueError("The remaining budget fraction requires [0, 1]")

    @classmethod
    def from_dict(cls, record):
        # Reference answers and verifier annotations never enter agent-visible state.
        return cls(**{key: record[key] for key in cls.__dataclass_fields__ if key in record})

    def canonical(self):
        return stable_text({"question": self.question, "observations": self.observations})

    def to_dict(self):
        return asdict(self)

    def features(self):
        previous = [set(re.findall(r"\w+", stable_text(value).lower())) for value in self.observations[:-1]]
        latest = set(re.findall(r"\w+", stable_text(self.observations[-1]).lower())) if self.observations else set()
        novelty = min((1 - len(latest & tokens) / max(1, len(latest | tokens)) for tokens in previous), default=1.0 if latest else 0.0)
        recent = [stable_text(value).lower() for value in (self.actions + self.proposals)[-8:]]
        repetition = (len(recent) - len(set(recent))) / max(1, len(recent))
        return [novelty, repetition, self.remaining_fraction]


@dataclass(frozen=True)
class Activity:
    activity_id: str
    category: str
    instruction: str


class ActivityPool:
    def __init__(self, records, categories):
        self.groups = {category: [] for category in categories}
        identifiers = set()
        for record in records:
            activity = record if isinstance(record, Activity) else Activity(**record)
            if activity.category not in self.groups or not activity.instruction.strip():
                raise ValueError("Activity category or instruction is invalid")
            if activity.activity_id in identifiers:
                raise ValueError("Activity identifiers require uniqueness")
            identifiers.add(activity.activity_id)
            self.groups[activity.category].append(activity)
        if any(not group for group in self.groups.values()):
            raise ValueError("Each configured activity category requires records")

    def sample(self, category, rng):
        # Sampling receives the category and RNG, without a scientific query.
        return rng.choice(self.groups[category])


def prepare_activities(records, seed=0, excluded_terms=()):
    mapping = {"brainstorming": "brainstorming", "creative_writing": "generation", "generation": "generation", "open_qa": "open_qa", "summarization": "summarization"}
    seen, activities = set(), []
    for record in records:
        category = mapping.get(record.get("category"))
        instruction = str(record.get("instruction", "")).strip()
        context = str(record.get("context", "")).strip()
        if not category or not instruction:
            continue
        text = instruction + ("\nContext: " + context if context else "")
        normalized = " ".join(text.casefold().split())
        if normalized in seen or any(term.casefold() in normalized for term in excluded_terms):
            continue
        if not context and re.search(r"\b(following|provided|above|below)\s+(text|passage|context|article)\b", instruction, re.I):
            continue
        seen.add(normalized)
        identifier = hashlib.sha256(normalized.encode()).hexdigest()[:20]
        # Source responses are excluded from the sampling pool.
        activities.append(asdict(Activity(identifier, category, text)))
    random.Random(seed).shuffle(activities)
    return activities, dict(Counter(record["category"] for record in activities))
