"""Answer evaluation: every candidate model answers the questions, a judge model grades them.

python -m ai_act_qa.answer_eval retrieve [--split dev]
  The passages for each question, once for all models -> reports/answers/<split>/retrieval.json
python -m ai_act_qa.answer_eval generate --mode binary --binary PATH [--models a,b] [--split dev]
  Each model answers every question with the prompt in `answers` (greedy decoding).
  -> reports/answers/<split>/<model>.jsonl (resumes: questions already answered are skipped)
python -m ai_act_qa.answer_eval judge --mode binary --binary PATH [--split dev]
  The judge grades each answer -> reports/answers/<split>/judged/<model>.jsonl
python -m ai_act_qa.answer_eval mac-latency [--models a,b] [--questions 5]
  Seconds per answer on this Mac (llama.cpp's Docker image, CPU) -> reports/mac_latency.json
python -m ai_act_qa.answer_eval report [--split dev]
  Metrics per model and the served model chosen by the rule -> reports/answers_<split>.json

--mode docker serves with llama.cpp's Docker image (the Mac); --mode binary with a local
llama-server (Kaggle). Model files go to $MODELS_DIR (default .cache/models); --delete-after
removes each one when its model is done (Kaggle's disk).

Deterministic metrics: declines (the fixed sentence), citations (do the cited numbers exist,
does a cited passage contain an expected one). Judged: correctness and groundedness.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

from ai_act_qa.answer import load_prompt, messages, read_answer
from ai_act_qa.artifacts import provenance, write_json
from ai_act_qa.chunk import build
from ai_act_qa.config import load_config, project_path
from ai_act_qa.eval_set import Question, load_questions
from ai_act_qa.llm import LlamaServer, chat, download
from ai_act_qa.parse import load_units
from ai_act_qa.retrieval_eval import overlaps
from ai_act_qa.search import Hit, Retriever

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "maxLength": 600},
        "correctness": {"enum": ["correct", "partial", "incorrect"]},
        "grounded": {"type": "boolean"},
    },
    "required": ["explanation", "correctness", "grounded"],
}
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def folder(split: str) -> Path:
    return project_path(f"reports/answers/{split}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", project_path(".cache/models")))


# -- retrieval ---------------------------------------------------------------------------------
def retrieve(split: str) -> dict[str, list[dict]]:
    cfg = load_config()
    canonical, units = load_units()
    retriever = Retriever(cfg["retrieval"])
    retriever.index(build(canonical, units, cfg["chunking"]))
    out = {}
    for q in load_questions(split):
        hits = retriever.search(q.question)
        out[q.id] = [{"chunk_id": h.chunk_id, "label": h.label, "text": h.text,
                      "spans": h.spans, "score": h.score} for h in hits]  # fmt: skip
    record = {"retrieval": cfg["retrieval"], "hits": out, "provenance": provenance()}
    write_json(f"reports/answers/{split}/retrieval.json", record)
    return out


def load_hits(split: str) -> dict[str, list[Hit]]:
    path = folder(split) / "retrieval.json"
    data = json.loads(path.read_text()) if path.exists() else {"hits": retrieve(split)}
    return {qid: [Hit(h["chunk_id"], h["label"], h["text"], [tuple(s) for s in h["spans"]],
                      h["score"]) for h in hits] for qid, hits in data["hits"].items()}  # fmt: skip


# -- generation --------------------------------------------------------------------------------
def server(args, cfg: dict, name: str, port: int = 8090) -> LlamaServer:
    spec = cfg["models"][name]
    path = download(spec, models_dir())
    return LlamaServer(path, args.mode, cfg["llama_cpp"]["context"], port=port,
                       binary=args.binary, threads=args.threads)  # fmt: skip


def answer_one(url: str, cfg: dict, prompt, q: Question, hits: list[Hit]) -> dict:
    acfg = cfg["answers"]
    reply = chat(url, messages(prompt, q.question, hits, acfg["decline"]), acfg["max_tokens"],
                 temperature=acfg["temperature"], seed=acfg["seed"],
                 thinking=acfg["thinking"])  # fmt: skip
    return {"id": q.id, **reply, **read_answer(reply["text"], len(hits), acfg["decline"])}


def generate(args) -> None:
    cfg = load_config()
    acfg = cfg["answers"]
    prompt = load_prompt(acfg["prompt"])
    questions = load_questions(args.split)
    if args.limit:
        questions = questions[: args.limit]
    hits = load_hits(args.split)
    for name in args.models or acfg["candidates"]:
        out = folder(args.split) / f"{name}.jsonl"
        done = {r["id"] for r in read_jsonl(out)}
        todo = [q for q in questions if q.id not in done]
        if not todo:
            print(f"{name}: all {len(questions)} answered")
            continue
        with server(args, cfg, name) as url:
            for q in todo:
                row = answer_one(url, cfg, prompt, q, hits[q.id])
                row.update(model=name, model_sha256=cfg["models"][name]["sha256"],
                           prompt=prompt.path, prompt_sha256=prompt.sha256,
                           llama_cpp=cfg["llama_cpp"]["tag"])  # fmt: skip
                append_jsonl(out, row)
                print(f"{name} {q.id}: {row['seconds']}s {row['text'][:80]!r}", flush=True)
        if args.delete_after:
            (models_dir() / cfg["models"][name]["file"]).unlink(missing_ok=True)


# -- judging -----------------------------------------------------------------------------------
def judge_messages(prompt, q: Question, hits: list[Hit], answer: str, units, canonical) -> list:
    if q.expected:
        texts = [(units[e].label, canonical[units[e].start : units[e].end]) for e in q.expected]
        expected = "\n\n".join(f"{label}:\n{text.strip()}" for label, text in texts)
    else:
        expected = "None: the Act does not answer this question."
    passages = "\n\n".join(f"[{i}] {h.text.strip()}" for i, h in enumerate(hits, 1))
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user.format(
            question=q.question, reference=q.answer, expected=expected, passages=passages,
            answer=answer.strip() or "(empty)")},
    ]  # fmt: skip


def judge(args) -> None:
    cfg = load_config()
    jcfg = {**cfg["judge"], "model": args.judge or cfg["judge"]["model"]}
    prompt = load_prompt(jcfg["prompt"])
    canonical, unit_list = load_units()
    units = {u.id: u for u in unit_list}
    questions = {q.id: q for q in load_questions(args.split)}
    hits = load_hits(args.split)
    todo = {}
    for path in sorted(folder(args.split).glob("*.jsonl")):
        out = folder(args.split) / "judged" / path.name
        done = {r["id"] for r in read_jsonl(out)}
        todo[out] = [r for r in read_jsonl(path) if r["id"] not in done and r["id"] in questions]
    if not any(todo.values()):
        print("everything is judged")
        return
    with server(args, cfg, jcfg["model"], port=8091) as url:
        for out, rows in todo.items():
            for row in rows:
                q = questions[row["id"]]
                reply = chat(url, judge_messages(prompt, q, hits[q.id], row["text"], units,
                                                 canonical),
                             jcfg["max_tokens"], schema=GRADE_SCHEMA)  # fmt: skip
                try:
                    grade = json.loads(reply["text"])
                except json.JSONDecodeError:
                    grade = {"error": reply["text"][:300]}
                append_jsonl(out, {"id": q.id, "model": row["model"], **grade,
                                   "judge": jcfg["model"], "judge_prompt_sha256": prompt.sha256,
                                   "seconds": reply["seconds"]})  # fmt: skip
                print(f"judged {row['model']} {q.id}: {grade.get('correctness')}", flush=True)
    if args.delete_after:
        (models_dir() / cfg["models"][jcfg["model"]]["file"]).unlink(missing_ok=True)


# -- Mac latency -------------------------------------------------------------------------------
def mac_latency(args) -> None:
    cfg = load_config()
    prompt = load_prompt(cfg["answers"]["prompt"])
    questions = [q for q in load_questions("dev") if q.answerable][: args.questions]
    hits = load_hits("dev")
    path = project_path("reports/mac_latency.json")
    results = json.loads(path.read_text())["models"] if path.exists() else {}
    names = args.models or [n for n in cfg["answers"]["candidates"] if cfg["models"][n]["mac"]]
    args.mode = "docker"
    for name in names:
        with server(args, cfg, name) as url:
            rows = [answer_one(url, cfg, prompt, q, hits[q.id]) for q in questions]
        seconds = [r["seconds"] for r in rows]
        results[name] = {"questions": [q.id for q in questions], "seconds": seconds,
                         "median_seconds": statistics.median(seconds),
                         "completion_tokens": [r["completion_tokens"] for r in rows]}  # fmt: skip
        print(f"{name}: median {statistics.median(seconds):.0f}s {seconds}", flush=True)
        record = {"threads": args.threads, "models": results, "provenance": provenance()}
        write_json("reports/mac_latency.json", record)


# -- report ------------------------------------------------------------------------------------
def summarise(rows: list[dict], grades: dict[str, dict], questions: dict[str, Question],
              hits: dict[str, list[Hit]], units: dict) -> dict:  # fmt: skip
    answerable = [r for r in rows if questions[r["id"]].answerable]
    unanswerable = [r for r in rows if not questions[r["id"]].answerable]
    answered = [r for r in answerable if not r["declined"]]

    def cites_expected(r: dict) -> bool:
        cited = [hits[r["id"]][n - 1] for n in r["citations"]]
        return any(overlaps(h.spans, units[e]) for h in cited for e in questions[r["id"]].expected)

    graded = [grades[r["id"]] for r in rows if "correctness" in grades.get(r["id"], {})]
    share = lambda xs, n: round(sum(xs) / n, 3) if n else None  # noqa: E731
    return {
        "questions": len(rows),
        "judged": len(graded),
        "score": share([SCORE[g["correctness"]] for g in graded], len(graded)),
        "correct": share([g["correctness"] == "correct" for g in graded], len(graded)),
        "partial": share([g["correctness"] == "partial" for g in graded], len(graded)),
        "grounded": share([g["grounded"] for g in graded], len(graded)),
        "declines_unanswerable": share([r["declined"] for r in unanswerable], len(unanswerable)),
        "declines_answerable": share([r["declined"] for r in answerable], len(answerable)),
        "answers_with_citation": share([bool(r["citations"]) for r in answered], len(answered)),
        "invalid_citations": share([bool(r["invalid"]) for r in rows], len(rows)),
        "cites_expected_passage": share([cites_expected(r) for r in answered], len(answered)),
        "truncated": share([r["finish_reason"] == "length" for r in rows], len(rows)),
        "median_completion_tokens": statistics.median(r["completion_tokens"] for r in rows),
    }


def select(summary: dict, cfg: dict, latency: dict) -> dict:
    """The rule in `selection`: fast enough on the Mac, then the best score, smaller first."""
    rule = cfg["selection"]
    eligible = [
        n
        for n in cfg["answers"]["candidates"]
        if cfg["models"][n]["mac"]
        and summary.get(n, {}).get("score") is not None
        and latency.get(n, {}).get("median_seconds", float("inf")) <= rule["mac_seconds"]
    ]
    if not eligible:
        return {"selected": None, "why": "no candidate has both a score and a Mac latency"}
    chosen = eligible[0]  # `candidates` is ordered from the smallest model
    for name in eligible[1:]:
        if (
            round(summary[name][rule["metric"]] - summary[chosen][rule["metric"]], 6)
            > rule["margin"]
        ):
            chosen = name
    return {"selected": chosen, "eligible": eligible, "rule": rule}


def report(args) -> None:
    cfg = load_config()
    canonical, unit_list = load_units()
    units = {u.id: u for u in unit_list}
    questions = {q.id: q for q in load_questions(args.split)}
    hits = load_hits(args.split)
    summary = {}
    for path in sorted(folder(args.split).glob("*.jsonl")):
        rows = [r for r in read_jsonl(path) if r["id"] in questions]
        judged = folder(args.split) / "judged" / path.name
        grades = {g["id"]: g for g in read_jsonl(judged)}
        summary[path.stem] = summarise(rows, grades, questions, hits, units)
    latency_path = project_path("reports/mac_latency.json")
    latency = json.loads(latency_path.read_text())["models"] if latency_path.exists() else {}
    selection = select(summary, cfg, latency) if args.split == "dev" else None
    write_json(f"reports/answers_{args.split}.json",
               {"split": args.split, "judge": cfg["judge"]["model"], "models": summary,
                "mac_latency": latency, "selection": selection,
                "provenance": provenance()})  # fmt: skip
    for name, s in summary.items():
        print(name, s)
    print("selection:", selection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["retrieve", "generate", "judge", "mac-latency",
                                            "report"])  # fmt: skip
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--models", type=lambda s: s.split(","), default=None)
    parser.add_argument("--mode", default="docker", choices=["docker", "binary"])
    parser.add_argument("--binary", default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--delete-after", action="store_true")
    parser.add_argument("--judge", default=None, help="another judge model (to test the code)")
    args = parser.parse_args()
    {"retrieve": lambda a: retrieve(a.split), "generate": generate, "judge": judge,
     "mac-latency": mac_latency, "report": report}[args.command](args)  # fmt: skip


if __name__ == "__main__":
    main()
