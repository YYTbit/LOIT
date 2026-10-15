import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_jsonl


def summarize_active(predictions_path, runs_path, output):
    predictions, runs = read_jsonl(predictions_path), read_jsonl(runs_path)
    grouped = defaultdict(list)
    for row in predictions:
        run_id = row.get("active_run_id")
        if not run_id or row.get("benchmark") != "agentideabench" or row.get("split") != "test":
            raise ValueError("Active prediction rows require run identifiers and the test split")
        if row.get("candidate_count") != 3 or not isinstance(row.get("hypothesis"), str) or not row["hypothesis"].strip():
            raise ValueError("Each Active prediction must contain one of three hypotheses")
        grouped[run_id].append(row)
    indexed = {row.get("active_run_id"): row for row in runs}
    if len(indexed) != len(runs) or set(indexed) != set(grouped):
        raise ValueError("Active prediction and trace run identifiers must match one-to-one")
    calls = []
    for run_id, rows in grouped.items():
        if len(rows) != 3 or {row.get("candidate_index") for row in rows} != {0, 1, 2}:
            raise ValueError(f"Active run {run_id} must have exactly three indexed hypotheses")
        run = indexed[run_id]
        trace = run.get("retrieval_trace", [])
        count = run.get("retrieval_calls")
        if (run.get("split") != "test" or count != len(trace) or type(count) is not int
                or not 0 <= count <= 10):
            raise ValueError(f"Active run {run_id} has inconsistent retrieval accounting")
        positions = run.get("processed_positions")
        incubation = run.get("incubation_positions", 0)
        if (type(positions) is not int or positions < 0 or type(incubation) is not int
                or incubation < 0 or incubation > positions):
            raise ValueError(f"Active run {run_id} has invalid decoder-position accounting")
        calls.append(count)
    if len(calls) != 40:
        raise ValueError("AgentIdeaBench Active artifacts require all 40 subfields")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    distribution = {str(value): Counter(calls)[value] for value in range(11)}
    summary = {"subfields": len(calls), "hypotheses": len(predictions),
               "retrieval_call_mean": statistics.fmean(calls),
               "retrieval_call_median": statistics.median(calls),
               "retrieval_call_distribution": distribution,
               "processed_positions": sum(int(row.get("processed_positions", 0)) for row in runs),
               "incubation_positions": sum(int(row.get("incubation_positions", 0)) for row in runs),
               "retrieved_papers": sum(len(row.get("retrieved_paper_ids", [])) for row in runs)}
    (output / "active_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
    _plot(distribution, output / "active_retrieval_calls.svg")
    return summary


def _plot(distribution, path):
    width, height, left, right, top, bottom = 760, 390, 70, 30, 55, 75
    chart_width, chart_height = width - left - right, height - top - bottom
    values = [distribution[str(index)] for index in range(11)]
    maximum = max(1, max(values))
    slot = chart_width / 11
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#202936">Observed Active retrieval calls</text>']
    for tick in range(6):
        value = maximum * tick / 5
        y = top + chart_height * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee8"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475467">{value:.0f}</text>')
    for index, value in enumerate(values):
        bar_width = slot * 0.62
        x = left + index * slot + (slot - bar_width) / 2
        bar_height = chart_height * value / maximum
        y = top + chart_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="2" fill="#3976a8"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#202936">{value}</text>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{top+chart_height+22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#202936">{index}</text>')
    parts.append(f'<text x="22" y="{top+chart_height/2}" transform="rotate(-90 22 {top+chart_height/2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#202936">Subfield count</text>')
    parts.append(f'<text x="{left+chart_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#202936">Retrieval calls per subfield</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
