import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = "paperId,title,abstract,year,citationCount"
REFERENCE_FIELDS = ("citedPaper.paperId,citedPaper.title,citedPaper.abstract,"
                    "citedPaper.year,citedPaper.citationCount,intents,contexts,isInfluential")
ACTIVE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"hypotheses": {"type": "array", "minItems": 3, "maxItems": 3,
                                    "items": {"type": "string", "minLength": 1}}},
    "required": ["hypotheses"],
    "additionalProperties": False,
}
COMMAND = re.compile(r"^(SEARCH|FETCH|FINAL)\s*:\s*(.*)$", re.IGNORECASE)
PAPER_ID = re.compile(r"^[A-Za-z0-9._:-]{1,80}$")


class SemanticScholarClient:
    def __init__(self, api_key=None, min_interval=1.5, opener=urlopen):
        self.api_key = api_key if api_key is not None else os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.min_interval = min_interval
        self.opener = opener
        self.last_request = 0.0

    def _get(self, path, params):
        wait = self.min_interval - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        url = f"{API_BASE}{path}?{urlencode(params)}"
        headers = {"User-Agent": "LOIT-AgentIdeaBench/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.last_request = time.monotonic()
        try:
            with self.opener(Request(url, headers=headers), timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def search(self, query, year_range=None, limit=10):
        limit = max(1, min(int(limit), 10))
        params = {"query": query, "fields": SEARCH_FIELDS, "limit": limit}
        if year_range:
            params["year"] = year_range
        data = self._get("/paper/search", params)
        if "error" in data:
            return data
        return [{"paperId": row.get("paperId"), "title": row.get("title"),
                 "abstract": (row.get("abstract") or "")[:1500], "year": row.get("year"),
                 "citationCount": row.get("citationCount") or 0}
                for row in (data.get("data") or []) if row]

    def fetch(self, paper_id):
        data = self._get(f"/paper/{paper_id}/references", {"fields": REFERENCE_FIELDS, "limit": 500})
        if "error" in data:
            return data
        references = []
        for edge in data.get("data") or []:
            edge = edge or {}
            paper = edge.get("citedPaper") or {}
            references.append({"paperId": paper.get("paperId"), "title": paper.get("title"),
                               "abstract": (paper.get("abstract") or "")[:1500],
                               "year": paper.get("year"), "citationCount": paper.get("citationCount") or 0,
                               "isInfluential": bool(edge.get("isInfluential"))})
        references.sort(key=lambda row: (row["isInfluential"], row["citationCount"]), reverse=True)
        return references[:15]


def _command(text):
    first = (text or "").strip().splitlines()
    if not first:
        return None, ""
    match = COMMAND.fullmatch(first[0].strip())
    return (match.group(1).upper(), match.group(2).strip()) if match else (None, "")


def _paper_block(rows, include_abstract):
    if isinstance(rows, dict) and "error" in rows:
        return json.dumps(rows, ensure_ascii=False)
    lines = []
    for index, row in enumerate((rows or [])[:10], 1):
        line = f"[{index}] paperId={row.get('paperId') or '?'} | year={row.get('year') or '?'} | {row.get('title') or '(untitled)'}"
        if include_abstract and index <= 5 and row.get("abstract"):
            line += f"\n    {row['abstract'][:350]}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no results)"


def _history(trace):
    lines = []
    for index, row in enumerate(trace):
        command = row["command"]
        label = "QUERY" if command == "SEARCH" else "PAPER"
        lines.append(f"{index + 1}. {label}: {row['argument']}")
        lines.append(_paper_block(row["results"], include_abstract=index >= len(trace) - 2))
    return "\n".join(lines)


def validate_active_tasks(tasks):
    from .data import ScientificState

    if len(tasks) != 40:
        raise ValueError("AgentIdeaBench Active evaluation requires all 40 subfields")
    domains, identifiers = set(), set()
    validated = []
    for task in tasks:
        if task.get("benchmark") != "agentideabench":
            raise ValueError("Active-track tasks require benchmark=agentideabench")
        state = ScientificState.from_dict(task.get("state", task))
        if state.split != "test" or not state.task_id or state.task_id in identifiers:
            raise ValueError("Active-track tasks require unique identifiers and the test split")
        subfield = task.get("subfield", state.question)
        if not isinstance(subfield, str) or not subfield.strip() or subfield in domains:
            raise ValueError("Active-track inputs require 40 distinct subfields")
        identifiers.add(state.task_id)
        domains.add(subfield)
        validated.append((task, state, subfield))
    return validated


def run_active_subfield(language, subfield, retrieval, seed=0, max_calls=10, cache=None):
    if not isinstance(subfield, str) or not subfield.strip():
        raise ValueError("AgentIdeaBench requires a nonempty Active-track subfield")
    if type(max_calls) is not int or not 1 <= max_calls <= 10:
        raise ValueError("AgentIdeaBench retrieval budget must lie in [1, 10]")
    trace, seen_ids = [], set()
    processed = 0
    for turn in range(max_calls):
        prompt = ("You are a scientific research agent. You receive only a research subfield. "
                  "Use Semantic Scholar to explore relevant literature. Output exactly one command "
                  "on the first line: SEARCH: <4-10 word query>, FETCH: <paperId from an earlier "
                  "search>, or FINAL: <stop exploration>. Each SEARCH or FETCH is one retrieval call. "
                  f"You have at most {max_calls} calls.\nSubfield: {subfield}\n"
                  f"Calls used: {len(trace)}\nRetrieval history:\n{_history(trace) or '(none)'}\n"
                  "Choose the next command.")
        completion = language.complete(prompt, language.config.action_tokens, seed + turn, cache=cache)
        processed += completion.processed_tokens
        action, argument = _command(completion.text)
        if action is None:
            raise ValueError("Active agent output must start with SEARCH:, FETCH:, or FINAL:")
        if action == "FINAL" or len(trace) == max_calls:
            break
        if action == "SEARCH":
            query = argument[:200].strip()
            results = retrieval.search(query, limit=10) if query else {"error": "SEARCH query is empty"}
            if isinstance(results, list):
                seen_ids.update(row.get("paperId") for row in results if row.get("paperId"))
        else:
            paper_id = argument.split()[0].strip("`\"'()[]{},.;") if argument else ""
            if not PAPER_ID.fullmatch(paper_id) or paper_id not in seen_ids:
                paper_id = paper_id[:80]
                results = {"error": "FETCH requires a paperId returned by an earlier SEARCH"}
            else:
                results = retrieval.fetch(paper_id)
        trace.append({"command": action, "argument": query if action == "SEARCH" else paper_id,
                      "results": results})

    final_prompt = ("Propose exactly three scientific hypotheses for this "
                    "AgentIdeaBench Active-track subfield, using the literature retrieved below. "
                    "Return only a JSON object with one key, "
                    "hypotheses, whose value is an array of exactly three strings. Do not add "
                    "scores or commentary.\n"
                    f"Subfield: {subfield}\nRetrieved literature:\n{_history(trace)}")
    final_limit = max(768, language.config.proposal_tokens * 4)
    if final_limit >= language.config.context_tokens:
        final_limit = language.config.context_tokens - 128
    if final_limit < 1:
        raise ValueError("The configured context does not leave room for Active-track outputs")
    hypotheses = None
    for attempt in range(2):
        completion = language.complete(final_prompt, final_limit, seed + max_calls + 1 + attempt,
                                       cache=cache, schema=ACTIVE_OUTPUT_SCHEMA)
        processed += completion.processed_tokens
        try:
            result = json.loads(completion.text)
            values = result.get("hypotheses") if isinstance(result, dict) else None
            if (isinstance(values, list) and len(values) == 3
                    and all(isinstance(value, str) and value.strip() for value in values)):
                hypotheses = [value.strip() for value in values]
                break
        except (ValueError, TypeError, AttributeError):
            pass
        final_prompt += ("\nThe previous response did not satisfy the exact three-item JSON "
                         "schema. Return a corrected JSON object.")
    if hypotheses is None:
        raise ValueError("Active-track generation did not produce three nonempty hypotheses")
    paper_ids = []
    for row in trace:
        if isinstance(row["results"], list):
            paper_ids.extend(item["paperId"] for item in row["results"] if item.get("paperId"))
    return {"subfield": subfield, "hypotheses": hypotheses, "retrieval_trace": trace,
            "retrieval_calls": len(trace), "retrieved_paper_ids": list(dict.fromkeys(paper_ids)),
            "processed_positions": processed, "seed": seed, "max_calls": max_calls}
