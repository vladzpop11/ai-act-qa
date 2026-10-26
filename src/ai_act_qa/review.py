"""Human review of the evaluation questions.

python -m ai_act_qa.review page
  -> eval/review.html: each question with the full text of its expected passages and the
  reference answer, and a verdict to record. Opened locally in a browser; progress is kept in
  the browser while reviewing, and "Download verdicts" saves review.json.
python -m ai_act_qa.review apply PATH
  Copies the downloaded verdicts into eval/review.json (committed), the record of which
  questions a person has checked. Questions marked for correction are fixed in
  eval/questions.yaml by hand, then reviewed again.
"""

import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_act_qa.artifacts import file_sha256
from ai_act_qa.config import project_path
from ai_act_qa.eval_set import QUESTIONS, load_questions
from ai_act_qa.parse import load_units

REVIEW = "eval/review.json"
VERDICTS = {
    "ok": "Correct",
    "passage": "Wrong or missing passage",
    "answer": "Reference answer wrong or incomplete",
    "question": "Question unclear or misleading",
}

STYLE = """
:root { --bg: #fbfaf7; --card: #ffffff; --text: #1d1d1b; --muted: #5f5d57; --line: #dedbd2;
        --accent: #2f5d8a; --ok: #2e7044; --warn: #9a5b13; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1b19; --card: #242421; --text: #ecebe6; --muted: #a9a79f; --line: #3b3a36;
          --accent: #8db7e0; --ok: #7cc493; --warn: #e0b06a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line);
         padding: 12px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
         z-index: 1; }
header h1 { font-size: 18px; margin: 0; flex: 1 1 auto; }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
.intro { color: var(--muted); }
article { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
          padding: 16px; margin: 0 0 16px; }
article.done { border-left: 4px solid var(--ok); }
article.flag { border-left: 4px solid var(--warn); }
.meta { color: var(--muted); font-size: 14px; }
h2 { font-size: 18px; margin: 4px 0 12px; }
h3 { font-size: 14px; margin: 12px 0 4px; color: var(--muted); text-transform: uppercase;
     letter-spacing: .04em; }
.passage { white-space: pre-wrap; border-left: 3px solid var(--line); padding: 4px 12px;
           margin: 6px 0; font-size: 15px; max-height: 320px; overflow: auto; }
.none { font-style: italic; }
fieldset { border: 0; padding: 0; margin: 12px 0 0; display: flex; flex-wrap: wrap; gap: 8px 16px; }
legend { font-weight: 600; margin-bottom: 4px; }
label { cursor: pointer; }
textarea { width: 100%; min-height: 56px; margin-top: 8px; font: inherit; color: var(--text);
           background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
button { font: inherit; background: var(--accent); color: var(--bg); border: 0; border-radius: 6px;
         padding: 8px 14px; cursor: pointer; }
button:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 3px solid var(--accent); outline-offset: 2px; }
"""

PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question review</title>
<style>
"""
    + STYLE
    + """</style>
</head>
<body>
<header>
  <h1>Question review <span class="meta" id="progress"></span></h1>
  <button type="button" id="download">Download verdicts</button>
</header>
<main>
<p class="intro">For each question, check that the passages shown really answer it, that the
reference answer matches them, and that the question is clear. An empty passage list means the
Act should not answer it. Your verdicts are kept in this browser as you go; when finished,
download them and send the file back. Questions file SHA-256: <code>__SHA__</code>.</p>
__CARDS__
</main>
<script>
const KEY = "ai-act-qa-review-__SHA__";
const total = __TOTAL__;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function paint() {
  let done = 0;
  document.querySelectorAll("article").forEach(card => {
    const v = state[card.id] || {};
    card.classList.toggle("done", v.verdict === "ok");
    card.classList.toggle("flag", !!v.verdict && v.verdict !== "ok");
    if (v.verdict) done += 1;
  });
  document.getElementById("progress").textContent = `${done} of ${total} reviewed`;
}
document.querySelectorAll("article").forEach(card => {
  const v = state[card.id] || {};
  card.querySelectorAll("input[type=radio]").forEach(r => {
    r.checked = r.value === v.verdict;
    r.addEventListener("change", () => {
      state[card.id] = {...(state[card.id] || {}), verdict: r.value}; save(); paint(); });
  });
  const note = card.querySelector("textarea");
  note.value = v.note || "";
  note.addEventListener("input", () => {
    state[card.id] = {...(state[card.id] || {}), note: note.value}; save(); });
});
document.getElementById("download").addEventListener("click", () => {
  const out = {questions_sha256: "__SHA__", reviewed_at: new Date().toISOString(), verdicts: state};
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "review.json"; a.click();
});
paint();
</script>
</body>
</html>
"""
)


def card(q, units: dict, canonical: str) -> str:
    esc = html.escape
    if q.expected:
        passages = "".join(
            f'<div class="passage"><strong>{esc(units[e].label)}</strong>\n'
            f"{esc(canonical[units[e].start : units[e].end].strip())}</div>"
            for e in q.expected
        )
    else:
        passages = '<p class="none">None: the Act should not answer this question.</p>'
    radios = "".join(
        f'<label><input type="radio" name="v-{q.id}" value="{key}"> {esc(text)}</label>'
        for key, text in VERDICTS.items()
    )
    return (
        f'<article id="{q.id}"><div class="meta">{q.id} · {esc(q.type)}</div>'
        f"<h2>{esc(q.question)}</h2><h3>Expected passages</h3>{passages}"
        f"<h3>Reference answer</h3><p>{esc(q.answer)}</p>"
        f"<fieldset><legend>Verdict</legend>{radios}</fieldset>"
        f'<label class="meta" for="n-{q.id}">Note (what to fix)</label>'
        f'<textarea id="n-{q.id}"></textarea></article>'
    )


def page() -> None:
    canonical, unit_list = load_units()
    units = {u.id: u for u in unit_list}
    questions = load_questions()
    sha = file_sha256(QUESTIONS)
    cards = "\n".join(card(q, units, canonical) for q in questions)
    out = PAGE.replace("__CARDS__", cards).replace("__SHA__", sha[:12])
    out = out.replace("__TOTAL__", str(len(questions)))
    path = project_path("eval/review.html")
    path.write_text(out, encoding="utf-8")
    print(f"wrote {path}")


def apply(source: str) -> None:
    data = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    if data["questions_sha256"] != file_sha256(QUESTIONS)[:12]:
        raise SystemExit("these verdicts are for another version of the questions file")
    ids = {q.id for q in load_questions()}
    verdicts = {k: v for k, v in data["verdicts"].items() if k in ids and v.get("verdict")}
    record = {
        "questions_sha256": file_sha256(QUESTIONS),
        "reviewed_at": data["reviewed_at"],
        "applied_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdicts": dict(sorted(verdicts.items())),
    }
    project_path(REVIEW).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    counts = {k: sum(v["verdict"] == k for v in verdicts.values()) for k in VERDICTS}
    print(f"{len(verdicts)} of {len(ids)} reviewed: {counts}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["apply"]:
        apply(sys.argv[2])
    else:
        page()
