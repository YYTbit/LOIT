from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os

from .data import ActivityPool, ScientificState, read_jsonl, stable_text
from .records import identify_records
from .storage import file_digest, software_versions, write_json


class CollectionRun:
    def __init__(self, output, identity, resume=False):
        self.output = Path(output)
        self.state_path = self.output.with_name(self.output.name + ".state.json")
        self.digest = hashlib.sha256()
        if resume:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["identity"] != identity:
                raise ValueError("Resume requires the same inputs, model, calibration, and execution settings")
            with self.output.open("rb") as handle:
                remaining = self.state["offset"]
                while remaining:
                    content = handle.read(min(1024 * 1024, remaining))
                    if not content:
                        raise ValueError("The saved prediction collection is incomplete")
                    self.digest.update(content)
                    remaining -= len(content)
            if self.digest.hexdigest() != self.state["sha256"]:
                raise ValueError("The committed prediction records have changed")
            # A partial last write is outside the committed task prefix.
            with self.output.open("r+b") as handle:
                handle.truncate(self.state["offset"])
        else:
            if self.output.exists() or self.state_path.exists():
                raise ValueError("An existing collection requires --resume or a new output")
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.touch(exist_ok=False)
            self.state = {"identity": identity, "next_task": 0, "records": 0, "offset": 0,
                          "sha256": self.digest.hexdigest(), "completed": False}
            write_json(self.state_path, self.state)

    def commit(self, index, record):
        if index != self.state["next_task"] or self.state["completed"]:
            raise ValueError("Collection commits follow the recorded task order")
        if record is not None:
            content = (stable_text(identify_records([record])[0]) + "\n").encode("utf-8")
            with self.output.open("ab") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            self.digest.update(content)
            self.state["offset"] += len(content)
            self.state["records"] += 1
        self.state.update(next_task=index + 1, sha256=self.digest.hexdigest())
        write_json(self.state_path, self.state)

    def finish(self):
        self.state["completed"] = True
        write_json(self.state_path, self.state)


def collect(language, args):
    from .benchmarks import load_environments
    from .runtime import Incubator, run_episode
    config = asdict(language.config)
    config.pop("device")
    identity = {"protocol": 1, "command": args.command, "configuration": config,
                "checkpoint_id": language.checkpoint_id, "seed": args.seed, "library_versions": software_versions(),
                "option_index": args.option_index,
                "inputs": {name: file_digest(getattr(args, name)) for name in
                           ["tasks", "activities", "calibration", "environments"] if getattr(args, name, None)}}
    if args.command == "run":
        identity.update(max_turns=args.max_turns, position_budget=args.position_budget,
                        training_rollouts=args.training_rollouts)
    run = CollectionRun(args.output, identity, args.resume)
    if run.state["completed"]:
        return run.state["records"]
    tasks = read_jsonl(args.tasks)
    incubator = Incubator(language, ActivityPool(read_jsonl(args.activities), language.config.categories))
    environments = load_environments(args.environments) if args.command == "run" else None
    for index in range(run.state["next_task"], len(tasks)):
        task = tasks[index]
        state = ScientificState.from_dict(task.get("state", task))
        if state.split == "unspecified" or not state.task_id:
            raise ValueError("Task records require an identifier and an explicit data split")
        seed = args.seed + index
        record = None
        if args.command == "run":
            record = run_episode(incubator, environments[task["benchmark"]], task, seed, args.max_turns,
                                 args.position_budget, args.training_rollouts, args.option_index)
            record["benchmark"] = task["benchmark"]
        elif args.command == "collect-associations":
            if state.split != "train":
                raise ValueError("Association collection requires designated training states")
            event = incubator.turn(state, seed, training=True, option_index=args.option_index).event
            if event["completed_incubation"]:
                record = {**event, "split": state.split}
        else:
            result = incubator.turn(state, seed, option_index=args.option_index)
            answer = language.complete("", language.config.action_tokens, seed + 1, cache=result.cache)
            record = {"task_id": state.task_id, "benchmark": task.get("benchmark"), "split": state.split,
                      "seed": seed, "answer": answer.text, "incubation": result.event,
                      "processed_positions": result.event["processed_positions"] + answer.processed_tokens}
        run.commit(index, record)
    run.finish()
    return run.state["records"]


def predictions(path):
    source = Path(path)
    state_path = source.with_name(source.name + ".state.json")
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state["completed"] or file_digest(source) != state["sha256"]:
            raise ValueError("Formal evaluation requires the completed, unchanged prediction collection")
    return read_jsonl(source)
