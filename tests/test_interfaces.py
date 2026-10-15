import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from loit.benchmarks import CommandSession, DiscoveryWorldSession, ScienceWorldSession, evaluate_collection
from loit.cli import main


def test_discoveryworld_keeps_scorecards_outside_observations(monkeypatch):
    class API:
        def __init__(self, threadID):
            self.calls = []
        def loadScenario(self, scenario, difficulty, seed, numUserAgents):
            self.calls.append((scenario, difficulty, seed, numUserAgents))
            return True
        def getAgentObservation(self, index):
            return {"ui": {"visible": "sensor"}, "scorecard": "SECRET", "vision": "image"}
        def listKnownActions(self):
            return {"MEASURE": {"arg1": "sensor"}}
        def additionalActionDescriptionString(self):
            return "Use a JSON action."
        def performAgentAction(self, index, action):
            self.calls.append(action)
            return {"success": True}
        def tick(self):
            self.calls.append("tick")
        def areTasksComplete(self):
            return True
        def getTaskScorecard(self):
            return [{"scoreNormalized": 0.6}, {"scoreNormalized": 0.8}]
    monkeypatch.setitem(sys.modules, "discoveryworld", ModuleType("discoveryworld"))
    monkeypatch.setitem(sys.modules, "discoveryworld.DiscoveryWorldAPI", SimpleNamespace(DiscoveryWorldAPI=API))
    session = DiscoveryWorldSession({"parameters": {"scenario": "Plant", "difficulty": "Normal", "world_seed": 2}}, 1, {})
    assert "SECRET" not in json.dumps(session.initial_observation)
    assert session.environment.calls[0] == ("Plant", "Normal", 2, 1)
    assert "MEASURE" in session.tools["actions"]
    observation, done = session.step('{"action":"MEASURE"}')
    assert done and observation["action_status"]["success"]
    assert session.score() == pytest.approx(0.7)
    assert not session.step("invalid JSON")[1]


def test_scienceworld_uses_released_action_interface(monkeypatch):
    class Environment:
        def __init__(self, serverPath, envStepLimit):
            self.closed = False
        def load(self, *args):
            self.loaded = args
        def reset(self):
            return "Visible lab", {"score": 0}
        def getPossibleActions(self):
            return ["look around", "measure OBJ"]
        def step(self, action):
            return "Measurement", 5, True, {"score": 80}
        def close(self):
            self.closed = True
    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Environment))
    session = ScienceWorldSession({"parameters": {"task_name": "measure", "variation": 4}}, 1, {})
    assert session.environment.loaded == ("measure", 4, "")
    assert session.step("measure sample") == ("Measurement", True)
    assert session.score() == 80
    session.close()
    assert session.environment.closed


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
    session = CommandSession([sys.executable, str(script), str(marker)], {}, 0, 5)
    assert session.initial_observation == "E" and session.tools == ["read"]
    session.close()
    assert marker.read_text() == "closed" and session.process.returncode == 0
