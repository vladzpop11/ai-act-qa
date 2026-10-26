"""Retrieval evaluation: does search return the passages that answer each question?

For each answerable question, a passage counts as found when one of the first k chunks contains
part of it (span overlap, so any chunking can be scored; a repeated lead-in does not count).

- recall@k: share of the expected passages found (averaged over questions);
- hit@k: share of questions with all their expected passages found;
- MRR: mean of 1 / rank of the first chunk containing an expected passage (0 if none in 20);
- words@k: mean words given to the model, the cost side of a larger k or larger chunks.

python -m ai_act_qa.retrieval_eval [--experiments] [--test]
  Default: the configured retrieval on the development questions -> reports/retrieval.json.
  --experiments: every setting in `retrieval_experiments` on the development questions
    -> reports/retrieval_experiments.json, with the setting the pre-registered rule selects.
  --test: the configured retrieval on the test questions, once, at the end of the project.
"""

import copy
import sys
import time

from ai_act_qa.artifacts import provenance, write_json
from ai_act_qa.chunk import build, words
from ai_act_qa.config import load_config
from ai_act_qa.eval_set import Question, load_questions
from ai_act_qa.parse import Unit, load_units
from ai_act_qa.search import Hit, Retriever

MRR_DEPTH = 20


def overlaps(spans: list[tuple[int, int]], unit: Unit) -> bool:
    return any(a < unit.end and unit.start < b for a, b in spans)


def score_question(q: Question, hits: list[Hit], units: dict[str, Unit], k: int) -> dict:
    expected = [units[e] for e in q.expected]
    found = [any(overlaps(h.spans, u) for h in hits[:k]) for u in expected]
    first = next((rank for rank, h in enumerate(hits, 1)
                  if any(overlaps(h.spans, u) for u in expected)), None)  # fmt: skip
    return {
        "id": q.id,
        "type": q.type,
        "recall": sum(found) / len(found),
        "hit": all(found),
        "rr": 1 / first if first else 0.0,
        "words": sum(words(h.text) for h in hits[:k]),
        "top": [h.chunk_id for h in hits[:k]],
    }


def evaluate(cfg: dict, questions: list[Question]) -> dict:
    """Index with `cfg` (chunking + retrieval), search every answerable question, score."""
    canonical, unit_list = load_units()
    units = {u.id: u for u in unit_list}
    chunks = build(canonical, unit_list, cfg["chunking"])
    retriever = Retriever(cfg["retrieval"])
    started = time.time()
    retriever.index(chunks)
    indexed = time.time() - started
    k = cfg["retrieval"]["k"]
    rows, seconds = [], []
    for q in questions:
        if not q.answerable:
            continue
        started = time.time()
        hits = retriever.search(q.question, k=max(k, MRR_DEPTH))
        seconds.append(time.time() - started)
        rows.append(score_question(q, hits, units, k))
    n = len(rows)
    return {
        "questions": n,
        "chunks": len(chunks),
        f"recall@{k}": round(sum(r["recall"] for r in rows) / n, 4),
        f"hit@{k}": round(sum(r["hit"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        f"words@{k}": round(sum(r["words"] for r in rows) / n, 1),
        "seconds_per_query": round(sum(seconds) / n, 2),
        "index_seconds": round(indexed, 1),
        "per_question": rows,
    }


def apply(cfg: dict, overrides: dict) -> dict:
    """The config with `overrides` ({"retrieval": {...}, "chunking": {...}}) applied."""
    out = copy.deepcopy(cfg)
    for section, values in overrides.items():
        out[section].update(values)
    return out


def select(results: dict, rule: dict) -> str:
    """The pre-registered rule: highest recall@k wins, but the baseline setting is kept unless
    another beats it by more than `margin`; ties go to fewer words given to the model."""
    metric = rule["metric"]
    baseline = rule["baseline"]
    best = max(results, key=lambda n: (results[n][metric], -results[n][rule["tie_break"]]))
    if results[best][metric] - results[baseline][metric] > rule["margin"]:
        return best
    return baseline


def main(argv: list[str]) -> None:
    cfg = load_config()
    split = "test" if "--test" in argv else "dev"
    questions = load_questions(split)
    if "--experiments" in argv:
        experiments = cfg["retrieval_experiments"]
        results = {}
        for name, overrides in experiments["settings"].items():
            results[name] = evaluate(apply(cfg, overrides or {}), questions)
            summary = {k: v for k, v in results[name].items() if k != "per_question"}
            print(f"{name}: {summary}", flush=True)
        chosen = select(results, experiments["rule"])
        write_json("reports/retrieval_experiments.json", {
            "split": split, "rule": experiments["rule"], "selected": chosen,
            "results": results, "provenance": provenance()})  # fmt: skip
        print(f"selected by the rule: {chosen}")
        return
    result = evaluate(cfg, questions)
    name = "reports/retrieval_test.json" if split == "test" else "reports/retrieval.json"
    write_json(name, {"split": split, **result, "provenance": provenance()})
    print({k: v for k, v in result.items() if k != "per_question"})


if __name__ == "__main__":
    main(sys.argv[1:])
