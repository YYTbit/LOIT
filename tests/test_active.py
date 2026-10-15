import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loit.active import SemanticScholarClient, run_active_subfield, validate_active_tasks
from loit.artifacts import summarize_active
from loit.cli import main


def hypothesis(prefix):
    return " ".join(f"{prefix}{index}" for index in range(80))


class FakeLanguage:
    config = SimpleNamespace(action_tokens=32, proposal_tokens=128, context_tokens=4096)

    def __init__(self, commands):
        self.commands = list(commands)
        self.prompts = []

    def complete(self, prompt, max_tokens, seed=0, schema=None, cache=None):
        self.prompts.append(prompt)
        if schema:
            text = json.dumps({"hypotheses": ["A brief mechanism.", "A distinct prediction.", "A third explanation."]})
        else:
            text = self.commands.pop(0)
        return SimpleNamespace(text=text, processed_tokens=4)


class FakeRetrieval:
    def __init__(self):
        self.calls = []

    def search(self, query, limit=10):
        self.calls.append(("SEARCH", query, limit))
        return [{"paperId": "paper-1", "title": "A paper", "abstract": "Evidence", "year": 2025}]

    def fetch(self, paper_id):
        self.calls.append(("FETCH", paper_id))
        return [{"paperId": "ref-1", "title": "A reference", "abstract": "Prior work", "year": 2020}]


def test_active_loop_uses_budget_records_trace_and_returns_three_hypotheses():
    language = FakeLanguage(["SEARCH: memory consolidation in science", "FETCH: paper-1"])
    retrieval = FakeRetrieval()
    result = run_active_subfield(language, "Scientific ideation", retrieval, seed=17, max_calls=2)
    assert len(result["hypotheses"]) == 3
    assert all(value for value in result["hypotheses"])
    assert result["retrieval_calls"] == 2
    assert result["retrieved_paper_ids"] == ["paper-1", "ref-1"]
    assert [row["command"] for row in result["retrieval_trace"]] == ["SEARCH", "FETCH"]
    assert retrieval.calls == [("SEARCH", "memory consolidation in science", 10), ("FETCH", "paper-1")]
    assert result["processed_positions"] == 12
    assert all("Scientific ideation" in prompt for prompt in language.prompts)


def test_active_fetch_requires_id_from_prior_search_and_still_charges_call():
    language = FakeLanguage(["FETCH: foreign-id"])
    retrieval = FakeRetrieval()
    result = run_active_subfield(language, "Biophysics", retrieval, max_calls=1)
    assert result["retrieval_calls"] == 1
    assert result["retrieval_trace"][0]["results"]["error"].startswith("FETCH requires")
    assert retrieval.calls == []


def test_active_task_input_requires_the_full_disjoint_test_set():
    tasks = [{"benchmark": "agentideabench", "task_id": f"id-{index}", "split": "test",
              "question": f"Subfield {index}"} for index in range(40)]
    assert len(validate_active_tasks(tasks)) == 40
    with pytest.raises(ValueError, match="40 subfields"):
        validate_active_tasks(tasks[:-1])
    tasks[1]["question"] = tasks[0]["question"]
    with pytest.raises(ValueError, match="distinct subfields"):
        validate_active_tasks(tasks)


def test_semantic_scholar_client_uses_released_paths_and_field_contract():
    requests = []

    class Response:
        def __init__(self, value):
            self.value = value
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(self.value).encode()

    def opener(request, timeout):
        requests.append((request.full_url, request.headers, timeout))
        if "/paper/search?" in request.full_url:
            return Response({"data": [{"paperId": "p1", "title": "Title", "abstract": "Text", "year": 2024}]})
        return Response({"data": [{"isInfluential": True,
                                     "citedPaper": {"paperId": "r1", "title": "Reference",
                                                    "abstract": "Text", "year": 2020,
                                                    "citationCount": 8}}]})

    client = SemanticScholarClient(api_key="fixture", min_interval=0, opener=opener)
    assert client.search("query", limit=20)[0]["paperId"] == "p1"
    assert client.fetch("p1")[0]["isInfluential"]
    assert len(requests) == 2
    assert "/paper/search?" in requests[0][0] and "limit=10" in requests[0][0]
    assert "/paper/p1/references?" in requests[1][0]
    assert requests[0][1]["X-api-key"] == "fixture"


def test_active_cli_writes_three_predictions_and_one_trace_per_subfield(tmp_path, monkeypatch, capsys):
    import loit.active
    import loit.cli

    config = tmp_path / "config.yaml"
    config.write_text("backbone: tiny\ndevice: cpu\nprecision: no\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"benchmark": "agentideabench", "task_id": f"id-{index}",
                                               "split": "test", "question": f"Field {index}"})
                                 for index in range(40)))
    activities = tmp_path / "activities.jsonl"
    activities.write_text("{}\n")
    class Retrieval:
        pass
    class Incubator:
        def __init__(self, language, activities):
            pass
        def turn(self, state, seed):
            return SimpleNamespace(cache="incubated-cache",
                                   event={"processed_positions": 5, "incubation_positions": 3,
                                          "option_index": 0})

    fake_language = SimpleNamespace(config=SimpleNamespace(backbone="fixture"),
                                    model=SimpleNamespace(backbone=SimpleNamespace(
                                        config=SimpleNamespace(_commit_hash="revision-1"))))
    monkeypatch.setattr(loit.cli, "language_from_args", lambda args: fake_language)
    monkeypatch.setattr(loit.cli, "pool_from_args", lambda args, config: object())
    monkeypatch.setattr("loit.runtime.Incubator", Incubator)
    monkeypatch.setattr(loit.active, "SemanticScholarClient", Retrieval)
    def generate(language, subfield, retrieval, seed, max_calls, cache=None):
        assert cache == "incubated-cache"
        return {"hypotheses": [hypothesis("a"), hypothesis("b"), hypothesis("c")],
                "retrieval_trace": [{"command": "SEARCH", "argument": subfield, "results": []}],
                "retrieval_calls": 1, "retrieved_paper_ids": [], "processed_positions": 7,
                "seed": seed, "max_calls": max_calls}
    monkeypatch.setattr(loit.active, "run_active_subfield", generate)
    output = tmp_path / "predictions.jsonl"
    main(["propose-agentideabench", "--config", str(config), "--activities", str(activities), "--tasks", str(tasks),
          "--output", str(output), "--max-retrieval-calls", "10"])
    predictions = [json.loads(row) for row in output.read_text().splitlines()]
    runs = [json.loads(row) for row in Path(str(output) + ".active_runs.jsonl").read_text().splitlines()]
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert len(predictions) == 120 and len(runs) == 40
    assert all(row["candidate_count"] == 3 and row["split"] == "test" for row in predictions)
    assert manifest["hypotheses"] == 120 and manifest["observed_retrieval_calls"] == 40
    assert manifest["incubation_positions"] == 120
    assert all(row["processed_positions"] == 12 for row in runs)
    assert json.loads(capsys.readouterr().out)["subfields"] == 40


def test_active_artifacts_validate_three_outputs_and_plot_observed_calls(tmp_path):
    predictions, runs = [], []
    for index in range(40):
        run_id = f"run-{index}"
        for candidate in range(3):
            predictions.append({"active_run_id": run_id, "benchmark": "agentideabench", "split": "test",
                                "candidate_index": candidate, "candidate_count": 3, "hypothesis": "idea"})
        count = index % 11
        runs.append({"active_run_id": run_id, "split": "test", "retrieval_calls": count,
                     "retrieval_trace": [{} for _ in range(count)], "retrieved_paper_ids": ["paper"],
                     "processed_positions": 20, "incubation_positions": 5})
    prediction_path, run_path = tmp_path / "predictions.jsonl", tmp_path / "runs.jsonl"
    prediction_path.write_text("\n".join(json.dumps(row) for row in predictions))
    run_path.write_text("\n".join(json.dumps(row) for row in runs))
    result = summarize_active(prediction_path, run_path, tmp_path / "artifacts")
    assert result["subfields"] == 40 and result["hypotheses"] == 120
    assert result["retrieval_call_mean"] == pytest.approx(4.65)
    assert result["processed_positions"] == 800 and result["incubation_positions"] == 200
    assert result["retrieved_papers"] == 40
    assert "Observed Active retrieval calls" in (tmp_path / "artifacts" / "active_retrieval_calls.svg").read_text()
