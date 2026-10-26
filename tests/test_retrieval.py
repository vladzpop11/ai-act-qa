"""Chunking, the evaluation questions and the retrieval metrics (no models needed)."""

import json

import pytest

from ai_act_qa.chunk import build, windows, words
from ai_act_qa.config import load_config, project_path
from ai_act_qa.eval_set import SPLIT, Question, check, load_questions, make_split
from ai_act_qa.parse import Unit, parse
from ai_act_qa.retrieval_eval import overlaps, score_question, select
from ai_act_qa.search import Hit


@pytest.fixture(scope="module")
def act():
    canonical, units = parse(project_path(load_config()["source"]["path"]).read_bytes())
    return canonical, units


@pytest.mark.parametrize("method", ["structural", "window"])
def test_chunks_cover_every_passage_within_size(act, method):
    canonical, units = act
    cfg = {**load_config()["chunking"], "method": method}
    chunks = build(canonical, units, cfg)
    assert len({c.id for c in chunks}) == len(chunks)
    covered = bytearray(len(canonical))
    for c in chunks:
        for a, b in c.spans:
            covered[a:b] = b"\x01" * (b - a)
    assert all(any(covered[u.start : u.end]) for u in units)
    limit = cfg["max_words"] + cfg["max_lead_words"] + 30 if method == "structural" else 200
    assert max(words(c.text) for c in chunks) <= limit


def test_lead_in_is_context_not_content(act):
    canonical, units = act
    chunks = {c.id: c for c in build(canonical, units, load_config()["chunking"])}
    by_id = {u.id: u for u in units}
    chunk = chunks["art5.1.f"]
    assert "The following AI practices shall be prohibited" in chunk.text
    assert not overlaps(chunk.spans, by_id["art5.1.a"])  # the lead-in is not point (a)
    assert overlaps(chunk.spans, by_id["art5.1.f"])


def test_windows_overlap_and_end_at_the_span():
    text = " ".join(f"w{i}" for i in range(25))
    spans = windows(text, (0, len(text)), size=10, overlap=5)
    assert [text[a:b].split()[0] for a, b in spans] == ["w0", "w5", "w10", "w15"]
    assert text[spans[-1][0] : spans[-1][1]].split()[-1] == "w24"


def test_questions_are_valid_and_split_is_fixed(act):
    questions = load_questions()
    assert check(questions, act[1]) == []
    unanswerable = sum(not q.answerable for q in questions) / len(questions)
    assert 0.05 <= unanswerable <= 0.15
    cfg = load_config()["evaluation"]
    stored = json.loads(project_path(SPLIT).read_text())
    assert stored["split"] == make_split(questions, cfg["test_share"], cfg["split_seed"])


def unit(uid, start, end):
    return Unit(uid, uid, "point", None, "", "", start, end)


def test_score_question_counts_overlap_rank_and_words():
    units = {"a": unit("a", 0, 10), "b": unit("b", 50, 60)}
    q = Question("q", "multi", "?", ["a", "b"], "")
    hits = [Hit("x", "", "one two", [(20, 30)], 1), Hit("y", "", "three", [(5, 15)], 0.5)]
    row = score_question(q, hits, units, k=2)
    assert row["recall"] == 0.5 and not row["hit"] and row["rr"] == 0.5 and row["words"] == 3


def test_selection_rule_keeps_baseline_within_margin():
    rule = {"metric": "recall@5", "baseline": "as_set", "margin": 0.04, "tie_break": "words@5"}
    results = {"as_set": {"recall@5": 0.80, "words@5": 500},
               "other": {"recall@5": 0.83, "words@5": 300}}  # fmt: skip
    assert select(results, rule) == "as_set"
    results["other"]["recall@5"] = 0.85
    assert select(results, rule) == "other"
