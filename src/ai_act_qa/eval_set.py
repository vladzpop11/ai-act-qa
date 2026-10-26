"""The evaluation questions (eval/questions.yaml) and their development/test split.

python -m ai_act_qa.eval_set split
  Assigns each question to `dev` or `test` once, at random within its type (seed fixed), and
  writes eval/split.json. Refuses if a split exists: the test questions stay the same for the
  whole project, so a test result is never chosen by re-splitting.
python -m ai_act_qa.eval_set check
  Checks the questions: known passage ids, unique ids, every question in the split.
"""

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

import yaml

from ai_act_qa.config import load_config, project_path
from ai_act_qa.parse import Unit

QUESTIONS = "eval/questions.yaml"
SPLIT = "eval/split.json"


@dataclass
class Question:
    id: str
    type: str
    question: str
    expected: list[str]
    answer: str
    reviewed: bool = False
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def load_questions(split: str | None = None) -> list[Question]:
    """All questions, or only those of `split` ("dev" or "test")."""
    raw = yaml.safe_load(project_path(QUESTIONS).read_text())["questions"]
    questions = [Question(**q) for q in raw]
    if split is None:
        return questions
    assignment = json.loads(project_path(SPLIT).read_text())["split"]
    return [q for q in questions if assignment[q.id] == split]


def make_split(questions: list[Question], test_share: float, seed: int) -> dict[str, str]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q.type].append(q.id)
    rng = random.Random(seed)
    split = {}
    for ids in by_type.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = round(len(ids) * test_share)
        split.update({i: "test" for i in ids[:n_test]})
        split.update({i: "dev" for i in ids[n_test:]})
    return dict(sorted(split.items()))


def check(questions: list[Question], units: list[Unit]) -> list[str]:
    known = {u.id for u in units}
    problems = [f"{q.id}: unknown passage {e}" for q in questions for e in q.expected
                if e not in known]  # fmt: skip
    ids = [q.id for q in questions]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    path = project_path(SPLIT)
    if path.exists():
        split = json.loads(path.read_text())["split"]
        problems += [f"{i} is not in {SPLIT}" for i in ids if i not in split]
    return problems


def main(command: str) -> None:
    cfg = load_config()["evaluation"]
    questions = load_questions()
    if command == "split":
        path = project_path(SPLIT)
        if path.exists():
            raise SystemExit(f"{SPLIT} exists: the split is fixed for the whole project")
        split = make_split(questions, cfg["test_share"], cfg["split_seed"])
        path.write_text(json.dumps({"seed": cfg["split_seed"], "test_share": cfg["test_share"],
                                    "split": split}, indent=1) + "\n")  # fmt: skip
        counts = {s: sum(v == s for v in split.values()) for s in ("dev", "test")}
        print(f"wrote {SPLIT}: {counts}")
    elif command == "check":
        from ai_act_qa.parse import load_units

        problems = check(questions, load_units()[1])
        print("\n".join(problems) or f"{len(questions)} questions OK")
        if problems:
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check")
