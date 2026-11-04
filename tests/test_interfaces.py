import json
import sys
from types import SimpleNamespace

import pytest

from loit.benchmarks import CommandEnvironment, evaluate_collection, load_environments
from loit.cli import main


def test_native_collection_evaluator_receives_whole_collection(monkeypatch):
    received = []
    def evaluate(records):
        received.append(records)
        return {"fluency": len(records)}
    monkeypatch.setitem(sys.modules, "native_fixture", SimpleNamespace(evaluate=evaluate))
    records = [{"task_id": "a"}, {"task_id": "b"}]
    assert evaluate_collection("liveideabench", records, "native_fixture:evaluate") == {"fluency": 2}
    assert received == [records]


def test_activity_preparation_cli_records_actual_counts_and_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"category": "brainstorming", "instruction": "Invent a method", "response": "PRIVATE"}]))
    output, manifest = tmp_path / "activities.jsonl", tmp_path / "manifest.json"
    main(["prepare-activities", "--input", str(source), "--output", str(output), "--manifest", str(manifest)])
    assert "PRIVATE" not in output.read_text()
    data = json.loads(manifest.read_text())
    assert data["count"] == 1 and len(data["sha256"]) == 64
    assert data["source"] == "local-input" and data["license"] == "unspecified"


def test_cli_help_is_available_without_model_loading(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "collect-associations" in capsys.readouterr().out


def test_native_worker_closes_resources_after_stdin_eof(tmp_path):
    marker, script = tmp_path / "closed.txt", tmp_path / "worker.py"
    script.write_text("import json, sys\nfrom pathlib import Path\n"
                      "for line in sys.stdin:\n"
                      "    json.loads(line)\n"
                      "    print(json.dumps({'observation': 'E', 'tools': ['read']}), flush=True)\n"
                      "Path(sys.argv[1]).write_text('closed')\n")
    environment = CommandEnvironment([sys.executable, str(script), str(marker)], timeout=5)
    session = environment.start({}, seed=0)
    assert session.initial_observation == "E" and session.tools == ["read"]
    session.close()
    assert marker.read_text() == "closed" and session.process.returncode == 0


def test_environment_configuration_loads_the_executable_example(tmp_path):
    manifest = tmp_path / "environments.json"
    manifest.write_text(json.dumps({"example": {"factory": "examples.environment:ExampleEnvironment"}}))
    environment = load_environments(manifest)["example"]
    session = environment.start({"observations": ["Recorded observation"]}, seed=17)
    assert environment.data_kind == "software_fixture"
    assert session.initial_observation == ["Recorded observation"]
    assert session.step("A proposal") == ("Proposal recorded.", True)
    session.close()


@pytest.mark.parametrize("processes,steps", [(1, 8), (2, 4), (4, 2)])
def test_training_cli_applies_global_batch_before_model_loading(tmp_path, monkeypatch, processes, steps):
    import loit.cli
    import loit.training
    from loit.config import Config

    config_path = tmp_path / "config.yaml"
    Config().save(config_path)
    loaded, trained = [], []

    def load(config, checkpoint):
        loaded.append((config.batch_size, config.global_batch_size, config.epochs))
        return SimpleNamespace(config=config)

    def train(language, stage, records, output, seed):
        trained.append((language.config.accumulation_steps(processes), stage, seed))

    monkeypatch.setattr(loit.cli, "load_language", load)
    monkeypatch.setattr(loit.training, "train", train)
    main(["train", "--config", str(config_path), "--stage", "focused",
          "--records", "training.jsonl", "--output", str(tmp_path / "output"),
          "--batch-size", "2", "--global-batch-size", "16", "--epochs", "1"])
    assert loaded == [(2, 16, 1)]
    assert trained == [(steps, "focused", 0)]


def test_training_cli_rejects_an_invalid_batch_before_model_loading(tmp_path, monkeypatch):
    import loit.cli
    from loit.config import Config

    config_path = tmp_path / "config.yaml"
    Config().save(config_path)
    loaded = []
    monkeypatch.setattr(loit.cli, "load_language", lambda *args: loaded.append(True))
    with pytest.raises(ValueError, match="global_batch_size"):
        main(["train", "--config", str(config_path), "--stage", "focused",
              "--records", "training.jsonl", "--output", str(tmp_path / "output"),
              "--global-batch-size", "0"])
    assert not loaded


def test_training_cli_rejects_the_removed_accumulation_override(capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", "config.yaml", "--stage", "focused",
              "--records", "training.jsonl", "--output", "output",
              "--gradient-accumulation", "8"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
