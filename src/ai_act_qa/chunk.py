"""Split the canonical text into chunks for retrieval, two ways:

- structural: one chunk per paragraph (or article, or top-level annex point) when it fits
  `max_words`; longer ones are split into their points, each carrying the lead-in sentence that
  introduces the list ("1. The following AI practices shall be prohibited:"); text with no
  structure left is cut into windows.
- window: fixed windows of `window_words` words with `overlap_words` overlap, within each
  article or annex.

Each chunk records its own spans of the canonical text (`spans`) and those of any lead-in it
repeats (`lead`): a chunk contains an expected passage when its own spans overlap the
passage's span, so a lead-in repeated for context never counts. With `header`, the text the
model sees starts with the article's citation, title and chapter.
"""

import re
from dataclasses import dataclass, field

from ai_act_qa.parse import Unit

Span = tuple[int, int]


@dataclass
class Chunk:
    id: str
    label: str  # citation of the unit it comes from
    unit: str  # unit id
    text: str
    spans: list[Span]
    lead: list[Span] = field(default_factory=list)


def words(text: str) -> int:
    return len(text.split())


def body_start(canonical: str, unit: Unit) -> int:
    """Where a unit's own text starts: after the heading line of an article or annex (the chunk
    header carries it)."""
    if unit.kind in {"article", "annex"}:
        return canonical.index("\n", unit.start) + 1
    return unit.start


def gaps(start: int, end: int, children: list[Unit]) -> tuple[list[Span], list[Span]]:
    """-> (the text before the first child, the text between and after children), as spans."""
    intro = [(start, children[0].start)] if children[0].start > start else []
    rest, cursor = [], children[0].end
    for child in children[1:]:
        if child.start > cursor:
            rest.append((cursor, child.start))
        cursor = max(cursor, child.end)
    if cursor < end:
        rest.append((cursor, end))
    return intro, rest


def windows(canonical: str, span: Span, size: int, overlap: int) -> list[Span]:
    """Word windows over `span`, as character spans."""
    bounds = [(m.start() + span[0], m.end() + span[0])
              for m in re.finditer(r"\S+", canonical[span[0]:span[1]])]  # fmt: skip
    if not bounds:
        return []
    out, step = [], max(1, size - overlap)
    for i in range(0, len(bounds), step):
        piece = bounds[i : i + size]
        out.append((piece[0][0], piece[-1][1]))
        if i + size >= len(bounds):
            break
    return out


class Chunker:
    def __init__(self, canonical: str, units: list[Unit], cfg: dict) -> None:
        self.canonical = canonical
        self.cfg = cfg
        self.children: dict[str, list[Unit]] = {}
        for unit in units:
            if unit.parent:
                self.children.setdefault(unit.parent, []).append(unit)
        self.roots = [u for u in units if u.parent is None]

    def text(self, spans: list[Span]) -> str:
        return "".join(self.canonical[a:b] for a, b in spans).strip()

    def header(self, root: Unit) -> str:
        if not self.cfg.get("header", True):
            return ""
        title = f"{root.label} - {root.title}" if root.title else root.label
        return f"{title} ({root.chapter})\n" if root.chapter else f"{title}\n"

    def make(self, cid: str, unit: Unit, root: Unit, spans: list[Span], lead: list[Span]) -> Chunk:
        body = self.text(lead + spans) if lead else self.text(spans)
        return Chunk(cid, unit.label, unit.id, self.header(root) + body, spans, lead)

    def chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        for root in self.roots:
            if self.cfg["method"] == "window":
                out += self.window_chunks(root)
            else:
                out += self.structural(root, root, lead=[])
        return out

    def window_chunks(self, root: Unit) -> list[Chunk]:
        start = self.canonical.index("\n", root.start) + 1
        spans = windows(self.canonical, (start, root.end), self.cfg["window_words"],
                        self.cfg["overlap_words"])  # fmt: skip
        return [self.make(f"{root.id}#w{i}", root, root, [s], []) for i, s in enumerate(spans)]

    def structural(self, unit: Unit, root: Unit, lead: list[Span]) -> list[Chunk]:
        start = body_start(self.canonical, unit)
        if start >= unit.end:
            return []
        whole = [(start, unit.end)]
        if words(self.text(lead + whole)) <= self.cfg["max_words"]:
            return [self.make(unit.id, unit, root, whole, lead)]
        children = self.children.get(unit.id, [])
        if not children:  # no structure left: windows, each with the lead-in
            spans = windows(self.canonical, whole[0], self.cfg["window_words"],
                            self.cfg["overlap_words"])  # fmt: skip
            return [self.make(f"{unit.id}#w{i}", unit, root, [s], lead)
                    for i, s in enumerate(spans)]  # fmt: skip
        intro, rest = gaps(start, unit.end, children)
        out: list[Chunk] = []
        if intro and words(self.text(intro)) > self.cfg["max_lead_words"]:
            out += self.structural_text(f"{unit.id}#intro", unit, root, intro[0], lead)
            child_lead = lead
        else:
            child_lead = lead + intro  # a short lead-in ("... the following:") goes with each part
        for child in children:
            out += self.structural(child, root, child_lead)
        for i, span in enumerate(rest):  # subparagraphs after the points
            out += self.structural_text(f"{unit.id}#t{i}", unit, root, span, lead)
        return out

    def structural_text(self, cid: str, unit: Unit, root: Unit, span: Span,
                        lead: list[Span]) -> list[Chunk]:  # fmt: skip
        """Text outside any child unit: one chunk, or windows if it is long."""
        if not self.text([span]):
            return []
        if words(self.text(lead + [span])) <= self.cfg["max_words"]:
            return [self.make(cid, unit, root, [span], lead)]
        spans = windows(self.canonical, span, self.cfg["window_words"], self.cfg["overlap_words"])
        return [self.make(f"{cid}.w{i}", unit, root, [s], lead) for i, s in enumerate(spans)]


def build(canonical: str, units: list[Unit], cfg: dict) -> list[Chunk]:
    return Chunker(canonical, units, cfg).chunks()
