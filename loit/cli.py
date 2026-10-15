import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .config import Config
from .data import ActivityPool, ScientificState, prepare_activities, read_jsonl, write_jsonl
from .storage import apply_calibration, load_language


def common(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--calibration")
    parser.add_argument("--seed", type=int, default=0)


def language_from_args(args):
    config = Config.load(args.config)
    if args.device:
        config.device = args.device
    if config.device.startswith("cuda") and "LOCAL_RANK" in os.environ:
        # Assign the local GPU before loading the shared backbone in distributed launches.
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        config.device = f"cuda:{rank}"
    if args.calibration:
        apply_calibration(config, args.calibration)
    torch.manual_seed(args.seed)
    return load_language(config, args.checkpoint)


def pool_from_args(args, config):
    return ActivityPool(read_jsonl(args.activities), config.categories)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Learned Off-Task Incubation Transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-activities")
    prepare.add_argument("--input")
    prepare.add_argument("--source", default="databricks/databricks-dolly-15k")
    prepare.add_argument("--provenance")
    prepare.add_argument("--license")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--exclude", nargs="*", default=["DiscoveryWorld", "ScienceWorld", "TOMATO-Star", "LiveIdeaBench", "AgentIdeaBench"])
    training = commands.add_parser("train")
    common(training)
    training.add_argument("--stage", choices=["focused", "initialize", "association", "return", "utility", "controller"], required=True)
    training.add_argument("--records", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--gradient-accumulation", type=int)
    for name in ["collect-associations", "propose", "run"]:
        command = commands.add_parser(name)
        common(command)
        command.add_argument("--activities", required=True)
        command.add_argument("--tasks", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--option-index", type=int)
        if name == "run":
            command.add_argument("--environments", required=True)
            command.add_argument("--max-turns", type=int, default=32)
            command.add_argument("--position-budget", type=int, default=32768)
            command.add_argument("--training-rollouts", action="store_true")
    calibration = commands.add_parser("calibrate")
    common(calibration)
    calibration.add_argument("--gate-records", required=True)
    calibration.add_argument("--paired-records", required=True)
    calibration.add_argument("--acceptance-floor", type=float, default=0.1)
    calibration.add_argument("--coverage-quantile", type=float, default=0.95)
    calibration.add_argument("--output", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--benchmark", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--evaluator", required=True)
    scoring.add_argument("--output", required=True)
    labels = commands.add_parser("attach-labels")
    labels.add_argument("--stage", choices=["association", "controller"], required=True)
    labels.add_argument("--records", required=True)
    labels.add_argument("--labels", required=True)
    labels.add_argument("--output", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--input", required=True)
    analysis.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-activities":
        if args.input:
            if Path(args.input).suffix == ".jsonl":
                source = read_jsonl(args.input)
            else:
                source = json.loads(Path(args.input).read_text(encoding="utf-8"))
        else:
            from datasets import load_dataset
            source = load_dataset(args.source, split="train")
        records, counts = prepare_activities(source, args.seed, args.exclude)
        if not records:
            raise ValueError("Activity filtering produced an empty pool")
        write_jsonl(args.output, records)
        source_name = args.provenance or ("local-input" if args.input else args.source)
        source_license = args.license or ("CC-BY-SA-3.0" if not args.input and args.source == "databricks/databricks-dolly-15k" else "unspecified")
        manifest = {"source": source_name, "license": source_license,
                    "categories": counts, "count": len(records), "seed": args.seed, "excluded_terms": args.exclude,
                    "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}
        write_json(args.manifest, manifest)
        print(json.dumps({"activities": len(records), "categories": counts}))
        return
    if args.command == "score":
        from .benchmarks import evaluate_collection
        write_json(args.output, evaluate_collection(args.benchmark, read_jsonl(args.predictions), args.evaluator))
        return
    if args.command == "attach-labels":
        from .records import attach_labels
        records = attach_labels(read_jsonl(args.records), read_jsonl(args.labels), args.stage)
        write_jsonl(args.output, records)
        return
    if args.command == "analyze":
        from .theory import escape_certificate
        values = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = escape_certificate(values["initial"], values["transition"], values["quality"], values["duration"],
                                    values["transition_samples"], values["return_samples"], values.get("delta", 0.05))
        write_json(args.output, result)
        return
    language = language_from_args(args)
    if args.command == "train":
        from .training import train
        for argument, field in [("epochs", "epochs"), ("batch_size", "batch_size"), ("gradient_accumulation", "gradient_accumulation")]:
            value = getattr(args, argument)
            if value is not None:
                setattr(language.config, field, value)
        language.config.__post_init__()
        train(language, args.stage, args.records, args.output, args.seed)
        return
    if args.command == "calibrate":
        from .calibration import gate_threshold, utility_margin
        values = {"gate_threshold": gate_threshold(read_jsonl(args.gate_records), args.acceptance_floor),
                  "utility_margin": utility_margin(language.model, read_jsonl(args.paired_records), args.coverage_quantile),
                  "coverage_quantile": args.coverage_quantile, "acceptance_floor": args.acceptance_floor}
        write_json(args.output, values)
        return
    from .runtime import Incubator, run_episode
    from .benchmarks import load_environments
    incubator = Incubator(language, pool_from_args(args, language.config))
    tasks = read_jsonl(args.tasks)
    environments = load_environments(args.environments) if args.command == "run" else None
    outputs = []
    for index, task in enumerate(tasks):
        state = ScientificState.from_dict(task.get("state", task))
        if state.split == "unspecified":
            raise ValueError("Task records require an explicit train, dev, validation, or test split")
        if not state.task_id:
            raise ValueError("Task records require a task identifier")
        seed = args.seed + index
        if args.command == "run":
            outputs.append(run_episode(incubator, environments[task["benchmark"]], task, seed, args.max_turns,
                                       args.position_budget, args.training_rollouts, args.option_index))
        elif args.command == "collect-associations":
            if state.split != "train":
                raise ValueError("Association collection requires designated training states")
            event = incubator.turn(state, seed, training=True, option_index=args.option_index).event
            if event["completed_incubation"]:
                event["split"] = state.split
                outputs.append(event)
        else:
            result = incubator.turn(state, seed, option_index=args.option_index)
            answer = language.complete("", language.config.action_tokens, seed + 1, cache=result.cache)
            outputs.append({"task_id": state.task_id, "benchmark": task.get("benchmark"), "split": state.split,
                            "seed": seed, "answer": answer.text, "incubation": result.event,
                            "processed_positions": result.event["processed_positions"] + answer.processed_tokens})
    from .records import identify_records
    write_jsonl(args.output, identify_records(outputs))
    print(json.dumps({"records": len(outputs), "command": args.command}))


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
