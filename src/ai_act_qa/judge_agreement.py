"""Checking the judge against a person, on development answers.

python -m ai_act_qa.judge_agreement sample
  -> eval/judge_sample.json (committed): which answers a person grades. Every question where the
  judge graded the two models of `judge_agreement.pair` differently (both answers: these decide
  the served model, and the judge shares a family with one of them), plus `random_per_model`
  answers of each other model, drawn with `seed`.
python -m ai_act_qa.judge_agreement page
  -> eval/judge_labels.html: the sampled answers in shuffled order, without the model or the
  judge's grade, each with the question, reference answer, expected passages and the passages
  the model was given. "Download labels" saves judge_labels.json.
python -m ai_act_qa.judge_agreement apply PATH
  Copies the downloaded labels into eval/judge_labels.json (committed).
python -m ai_act_qa.judge_agreement report
  -> reports/judge_agreement.json: agreement and Cohen's kappa between person and judge, on the
  random answers and on the pair's; each model's score on the pair's questions by person and by
  judge; and how much more lenient the judge is than the person, per model family.
"""

import html
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_act_qa.answer import read_answer
from ai_act_qa.answer_eval import SCORE, folder, load_hits, read_jsonl
from ai_act_qa.artifacts import file_sha256, provenance, write_json
from ai_act_qa.config import load_config, project_path
from ai_act_qa.eval_set import load_questions
from ai_act_qa.parse import load_units
from ai_act_qa.review import STYLE

SAMPLE = "eval/judge_sample.json"
LABELS = "eval/judge_labels.json"
GRADES = {"correct": "Correct", "partial": "Partial", "incorrect": "Incorrect"}

RUBRIC = """<ul>
<li><strong>Correct</strong>: gives the key facts of the reference answer (numbers, dates,
conditions, exceptions that change the answer) and states nothing that contradicts the
reference or the expected passages.</li>
<li><strong>Partial</strong>: on the right track but leaves out a key fact or condition, or adds
a minor inaccuracy, and nothing in it is seriously wrong.</li>
<li><strong>Incorrect</strong>: wrong, contradicts the reference, answers a different question,
or declines although the reference answer gives an answer (even if the passages the service
was given do not contain it).</li>
<li>If the reference says the Act does not answer the question, a reply saying so is
<strong>correct</strong> and any other answer is <strong>incorrect</strong>.</li>
</ul>"""

PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer labelling</title>
<style>
"""
    + STYLE
    + """
details { margin: 8px 0; } summary { cursor: pointer; color: var(--accent); }
.answer { white-space: pre-wrap; background: var(--bg); border: 1px solid var(--line);
          border-radius: 6px; padding: 8px 12px; }
</style>
</head>
<body>
<header>
  <h1>Answer labelling <span class="meta" id="progress"></span></h1>
  <button type="button" id="download">Download labels</button>
</header>
<main>
<p class="intro">Grade each answer against the reference answer and the expected passages, as
the judge model was asked to. The model and the judge's grade are hidden, and the answers are
shuffled. Your labels are kept in this browser as you go; when finished, download them and
send the file back. Sample SHA-256: <code>__SHA__</code>.</p>
__RUBRIC__
__CARDS__
</main>
<script>
const KEY = "ai-act-qa-labels-__SHA__";
const total = __TOTAL__;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function paint() {
  let done = 0;
  document.querySelectorAll("article").forEach(card => {
    const v = state[card.id] || {};
    card.classList.toggle("done", !!v.label);
    if (v.label) done += 1;
  });
  document.getElementById("progress").textContent = `${done} of ${total} labelled`;
}
document.querySelectorAll("article").forEach(card => {
  const v = state[card.id] || {};
  card.querySelectorAll("input[type=radio]").forEach(r => {
    r.checked = r.value === v.label;
    r.addEventListener("change", () => {
      state[card.id] = {...(state[card.id] || {}), label: r.value}; save(); paint(); });
  });
  const note = card.querySelector("textarea");
  note.value = v.note || "";
  note.addEventListener("input", () => {
    state[card.id] = {...(state[card.id] || {}), note: note.value}; save(); });
});
document.getElementById("download").addEventListener("click", () => {
  const out = {sample_sha256: "__SHA__", labelled_at: new Date().toISOString(), labels: state};
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "judge_labels.json"; a.click();
});
paint();
</script>
</body>
</html>
"""
)


def answers(split: str = "dev") -> tuple[dict, dict]:
    """-> ({model: {question id: answer row}}, {model: {question id: judge grade}})."""
    rows, grades = {}, {}
    for path in sorted(folder(split).glob("*.jsonl")):
        rows[path.stem] = {r["id"]: r for r in read_jsonl(path)}
        grades[path.stem] = {g["id"]: g for g in read_jsonl(folder(split) / "judged" / path.name)}
    return rows, grades


def sample() -> None:
    cfg = load_config()["judge_agreement"]
    rows, grades = answers()
    a, b = cfg["pair"]
    differ = sorted(
        q for q in grades[a] if grades[a][q]["correctness"] != grades[b][q]["correctness"]
    )
    items = [{"model": m, "id": q, "set": "pair"} for q in differ for m in (a, b)]
    rng = random.Random(cfg["seed"])
    for model in sorted(set(rows) - {a, b}):
        for q in rng.sample(sorted(rows[model]), cfg["random_per_model"]):
            items.append({"model": model, "id": q, "set": "random"})
    rng.shuffle(items)
    for n, item in enumerate(items, 1):
        item["item"] = f"a{n:02d}"
    write_json(SAMPLE, {"pair": cfg["pair"], "seed": cfg["seed"], "items": items,
                        "provenance": provenance()})  # fmt: skip
    print(f"{len(items)} answers: {len(differ)} questions where {a} and {b} differ, "
          f"{len(items) - 2 * len(differ)} random")  # fmt: skip


def card(item: dict, row: dict, q, hits, units: dict, canonical: str) -> str:
    esc = html.escape
    if q.expected:
        expected = "".join(
            f'<div class="passage"><strong>{esc(units[e].label)}</strong>\n'
            f"{esc(canonical[units[e].start : units[e].end].strip())}</div>"
            for e in q.expected
        )
    else:
        expected = '<p class="none">None: the Act does not answer this question.</p>'
    given = "".join(f'<div class="passage"><strong>[{i}]</strong> {esc(h.text.strip())}</div>'
                    for i, h in enumerate(hits, 1))  # fmt: skip
    radios = "".join(
        f'<label><input type="radio" name="g-{item["item"]}" value="{key}"> {text}</label>'
        for key, text in GRADES.items()
    )
    return (
        f'<article id="{item["item"]}"><div class="meta">{item["item"]} · {esc(q.id)}</div>'
        f"<h2>{esc(q.question)}</h2><h3>Reference answer</h3><p>{esc(q.answer)}</p>"
        f"<h3>Expected passages</h3>{expected}"
        f"<details><summary>Passages the service was given</summary>{given}</details>"
        f'<h3>Answer</h3><div class="answer">{esc(row["text"].strip())}</div>'
        f"<fieldset><legend>Grade</legend>{radios}</fieldset>"
        f'<label class="meta" for="n-{item["item"]}">Note (optional)</label>'
        f'<textarea id="n-{item["item"]}"></textarea></article>'
    )


def page() -> None:
    canonical, unit_list = load_units()
    units = {u.id: u for u in unit_list}
    questions = {q.id: q for q in load_questions("dev")}
    hits = load_hits("dev")
    rows, _ = answers()
    items = json.loads(project_path(SAMPLE).read_text())["items"]
    cards = "\n".join(card(i, rows[i["model"]][i["id"]], questions[i["id"]], hits[i["id"]], units,
                           canonical) for i in items)  # fmt: skip
    sha = file_sha256(SAMPLE)[:12]
    out = PAGE.replace("__CARDS__", cards).replace("__RUBRIC__", RUBRIC)
    out = out.replace("__SHA__", sha).replace("__TOTAL__", str(len(items)))
    path = project_path("eval/judge_labels.html")
    path.write_text(out, encoding="utf-8")
    print(f"wrote {path}")


def apply(source: str) -> None:
    data = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    if data["sample_sha256"] != file_sha256(SAMPLE)[:12]:
        raise SystemExit("these labels are for another sample")
    items = {i["item"] for i in json.loads(project_path(SAMPLE).read_text())["items"]}
    labels = {k: v for k, v in data["labels"].items() if k in items and v.get("label")}
    record = {
        "sample_sha256": file_sha256(SAMPLE),
        "labelled_at": data["labelled_at"],
        "applied_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels": dict(sorted(labels.items())),
    }
    project_path(LABELS).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(labels)} of {len(items)} labelled")


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa of two raters over the same items."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(sum(a == k for a, _ in pairs) * sum(b == k for _, b in pairs)
                   for k in GRADES) / n**2  # fmt: skip
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else None


def compare(items: list[dict]) -> dict:
    pairs = [(i["person"], i["judge"]) for i in items]
    return {
        "answers": len(items),
        "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
        "kappa": kappa(pairs),
        "agreement_rubric": round(
            sum(i["person"] == i["judge_rubric"] for i in items) / len(items), 3
        )
        if items
        else None,  # fmt: skip
    }


def report() -> None:
    cfg = load_config()
    sample_data = json.loads(project_path(SAMPLE).read_text())
    labels = json.loads(project_path(LABELS).read_text())["labels"]
    if json.loads(project_path(LABELS).read_text())["sample_sha256"] != file_sha256(SAMPLE):
        raise SystemExit("the labels are for another sample")
    questions = {q.id: q for q in load_questions("dev")}
    rows, grades = answers()
    decline = cfg["answers"]["decline"]
    items = []
    for i in sample_data["items"]:
        if i["item"] not in labels:
            continue
        row, q = rows[i["model"]][i["id"]], questions[i["id"]]
        judge = grades[i["model"]][i["id"]]["correctness"]
        declined = read_answer(row["text"], 5, decline)["declined"] and q.answerable
        items.append({**i, "person": labels[i["item"]]["label"], "judge": judge,
                      "judge_rubric": "incorrect" if declined else judge,
                      "family": cfg["models"][i["model"]]["family"]})  # fmt: skip
    a, b = sample_data["pair"]
    pair = [i for i in items if i["set"] == "pair"]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    pair_scores = {m: {"person": mean([SCORE[i["person"]] for i in pair if i["model"] == m]),
                       "judge": mean([SCORE[i["judge_rubric"]] for i in pair if i["model"] == m])}
                   for m in (a, b)}  # fmt: skip
    leniency = {f: mean([SCORE[i["judge_rubric"]] - SCORE[i["person"]]
                         for i in items if i["family"] == f])
                for f in sorted({i["family"] for i in items})}  # fmt: skip
    write_json("reports/judge_agreement.json", {
        "judge": cfg["judge"]["model"],
        "labelled": len(items), "sampled": len(sample_data["items"]),
        "random": compare([i for i in items if i["set"] == "random"]),
        "pair": compare(pair), "all": compare(items),
        "pair_scores": pair_scores,
        "judge_minus_person_by_family": leniency,
        "disagreements": [{k: i[k] for k in ("item", "model", "id", "person", "judge")}
                          for i in items if i["person"] != i["judge_rubric"]],
        "provenance": provenance(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in json.loads(
        project_path("reports/judge_agreement.json").read_text()).items()
        if k not in ("disagreements", "provenance")}, indent=1))  # fmt: skip


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "page"
    if command == "apply":
        apply(sys.argv[2])
    else:
        {"sample": sample, "page": page, "report": report}[command]()
