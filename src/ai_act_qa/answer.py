"""Answering a question from retrieved passages: the prompt, and reading the answer back.

The prompt (prompts/answer_v1.txt) numbers the passages [1] to [k] and asks for a citation
after each statement, or for one fixed sentence when the passages do not answer. Reading the
answer is deterministic: which passage numbers it cites (and whether they exist), and whether
it declined.
"""

import re
from dataclasses import dataclass

from ai_act_qa.artifacts import file_sha256
from ai_act_qa.config import project_path
from ai_act_qa.search import Hit

CITATION = re.compile(r"\[(\d+)\]")


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


def read_answer(text: str, k: int, decline: str) -> dict:
    """-> {"citations": valid passage numbers, "invalid": numbers outside 1..k, "declined"}."""
    numbers = [int(n) for n in CITATION.findall(text)]
    return {
        "citations": sorted({n for n in numbers if 1 <= n <= k}),
        "invalid": sorted({n for n in numbers if not 1 <= n <= k}),
        "declined": normalise(decline).rstrip(".") in normalise(text),
    }
