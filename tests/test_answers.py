"""Reading answers, the deterministic answer metrics and the served-model rule (no model)."""

from ai_act_qa.answer import load_prompt, messages, read_answer
from ai_act_qa.answer_eval import select, summarise
from ai_act_qa.config import load_config
from ai_act_qa.eval_set import Question
from ai_act_qa.parse import Unit
from ai_act_qa.search import Hit

DECLINE = "The AI Act passages provided do not answer this question."


def test_citations_are_read_and_checked_against_the_passages_given():
    read = read_answer("Banned [1]. Except for safety [2][2] and [7].", k=5, decline=DECLINE)
    assert read == {"citations": [1, 2], "invalid": [7], "declined": False, "exact_decline": False}


def test_decline_is_recognised_whatever_the_spacing_and_case():
    text = "the AI Act passages provided do not\nanswer this question"
    assert read_answer(text, k=5, decline=DECLINE)["exact_decline"]
    assert not read_answer("It is not allowed [1].", k=5, decline=DECLINE)["declined"]


def test_declines_in_other_words_count_but_claims_about_the_act_do_not():
    declines = [
        "The provided passages do not contain information about relief on fines.",
        "The provided text does not explicitly state whether exam tools are high-risk.",
        "The cost is not explicitly defined in the provided text.",
        "The provided passages do not answer the question.",
    ]  # all seen in the first dev run
    for text in declines:
        read = read_answer(text, k=5, decline=DECLINE)
        assert read["declined"] and not read["exact_decline"], text
    answers = [
        "The AI Act does not provide any tax incentives for AI start-ups.",
        "No, it is not allowed [1]. The provided passages do not mention other cases.",
        f"High-risk systems must comply [2]. {DECLINE}",
    ]  # a claim about the Act; declines only after an answer
    for text in answers:
        assert not read_answer(text, k=5, decline=DECLINE)["declined"], text


def test_prompt_numbers_passages_and_states_the_decline_sentence():
    prompt = load_prompt(load_config()["answers"]["prompt"])
    hits = [Hit("a", "", "Article 5 text", [(0, 1)], 1.0), Hit("b", "", "Annex III text", [], 0)]
    system, user = messages(prompt, "Is X banned?", hits, DECLINE)
    assert DECLINE in system["content"] and "{" not in system["content"]
    assert "[1] Article 5 text" in user["content"] and "[2] Annex III text" in user["content"]
    assert user["content"].endswith("Question: Is X banned?")


def test_summary_metrics():
    units = {"u": Unit("u", "u", "point", None, "", "", 0, 10)}
    questions = {"a": Question("a", "t", "?", ["u"], "r"), "b": Question("b", "t", "?", [], "r")}
    hits = {"a": [Hit("x", "", "t", [(20, 30)], 1), Hit("y", "", "t", [(5, 8)], 1)], "b": []}
    rows = [
        {"id": "a", "text": "Yes [2].", "finish_reason": "stop", "completion_tokens": 40},
        {"id": "b", "text": DECLINE, "finish_reason": "stop", "completion_tokens": 12},
    ]  # fmt: skip
    grades = {"a": {"correctness": "partial", "grounded": True},
              "b": {"correctness": "correct", "grounded": True}}  # fmt: skip
    s = summarise(rows, grades, questions, hits, units, DECLINE)
    assert s["score"] == 0.75 and s["declines_unanswerable"] == 1.0
    assert s["declines_answerable"] == 0.0 and s["cites_expected_passage"] == 1.0
    assert s["declines_with_fixed_sentence"] == 1.0


def test_score_counts_a_decline_on_an_answerable_question_as_incorrect():
    units = {"u": Unit("u", "u", "point", None, "", "", 0, 10)}
    questions = {"a": Question("a", "t", "?", ["u"], "r")}
    hits = {"a": [Hit("x", "", "t", [(20, 30)], 1)]}  # retrieval missed the passage
    rows = [{"id": "a", "text": "The provided passages do not contain this.",
             "finish_reason": "stop", "completion_tokens": 9}]  # fmt: skip
    grades = {"a": {"correctness": "correct", "grounded": True}}  # the judge excused it
    s = summarise(rows, grades, questions, hits, units, DECLINE)
    assert s["score"] == 0.0 and s["score_judge"] == 1.0
    assert s["declines_answerable"] == 1.0 and s["declines_with_fixed_sentence"] == 0.0


def test_selection_prefers_the_smaller_model_within_the_margin_and_needs_mac_speed():
    cfg = {
        "selection": {"metric": "score", "margin": 0.05, "mac_seconds": 150},
        "answers": {"candidates": ["small", "mid", "big"]},
        "models": {"small": {"mac": True}, "mid": {"mac": True}, "big": {"mac": True}},
    }
    summary = {"small": {"score": 0.60}, "mid": {"score": 0.64}, "big": {"score": 0.80}}
    latency = {"small": {"median_seconds": 40}, "mid": {"median_seconds": 90},
               "big": {"median_seconds": 400}}  # fmt: skip
    assert select(summary, cfg, latency)["selected"] == "small"  # big too slow, mid within margin
    summary["mid"]["score"] = 0.70
    assert select(summary, cfg, latency)["selected"] == "mid"
    assert select(summary, cfg, {})["selected"] is None
