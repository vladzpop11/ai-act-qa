"""Answering a question from retrieved passages: the prompt, and reading the answer back.

The prompt (prompts/answer_v1.txt) numbers the passages [1] to [k] and asks for a citation
after each statement, or for one fixed sentence when the passages do not answer. Reading the
answer is deterministic: which passage numbers it cites (and whether they exist), and whether
it declined. Models often decline in their own words ("The provided passages do not contain
..."), so a decline is an answer whose first sentence is the fixed sentence or says that the
passages given do not answer. "The AI Act does not ..." is a claim about the Act, so an
answer, and so is an answer that adds the fixed sentence after answering.
"""

import re
from dataclasses import dataclass

from ai_act_qa.artifacts import file_sha256
from ai_act_qa.config import project_path
from ai_act_qa.search import Hit

CITATION = re.compile(r"\[(\d+)\]")
_VERBS = "answer|contain|specify|mention|state|define|provide|include|address|say|cover"
DECLINES = [
    # "The provided passages do not (explicitly) state ..."
    re.compile(r"\b(passages?|text|provisions|excerpts?)\b[^.]{0,60}?\b(do|does)\s+not\s+"
               rf"(\w+ly\s+)?({_VERBS})\b", re.I),
    # "... is not (explicitly) defined in the provided text"
    re.compile(r"\bnot\s+(\w+ly\s+)?(answered|addressed|defined|mentioned|specified|stated|"
               r"covered|included|contained)\s+in\s+the\s+(provided|given)\s+"
               r"(passages?|text|provisions|excerpts?)\b", re.I),
]  # fmt: skip


@dataclass
class Prompt:
    path: str
    system: str
    user: str
    sha256: str


def load_prompt(relative: str) -> Prompt:
    """A prompt file: '=== system' and '=== user' sections with {placeholders}."""
    text = project_path(relative).read_text(encoding="utf-8")
    sections = dict(re.findall(r"^=== (\w+)\n(.*?)(?=^=== |\Z)", text, flags=re.S | re.M))
    return Prompt(relative, sections["system"].strip(), sections["user"].strip(),
                  file_sha256(relative))  # fmt: skip


def passages_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.text.strip()}" for i, h in enumerate(hits, 1))


def messages(prompt: Prompt, question: str, hits: list[Hit], decline: str) -> list[dict]:
    return [
        {"role": "system", "content": prompt.system.format(decline=decline)},
        {
            "role": "user",
            "content": prompt.user.format(passages=passages_block(hits), question=question),
        },  # fmt: skip
    ]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def read_answer(text: str, k: int, decline: str) -> dict:
    """-> {"citations": valid passage numbers, "invalid": numbers outside 1..k,
    "declined": the fixed sentence or in other words, "exact_decline": the fixed sentence}."""
    numbers = [int(n) for n in CITATION.findall(text)]
    first = first_sentence(text)
    exact = normalise(decline).rstrip(".") in normalise(first)
    return {
        "citations": sorted({n for n in numbers if 1 <= n <= k}),
        "invalid": sorted({n for n in numbers if not 1 <= n <= k}),
        "declined": exact or any(p.search(first) for p in DECLINES),
        "exact_decline": exact,
    }
