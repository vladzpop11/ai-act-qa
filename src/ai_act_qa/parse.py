"""Parse the consolidated AI Act (XHTML) into citable units over one canonical text.

The canonical text is the enacting terms and annexes as plain lines ("Article 5 - Prohibited AI
practices", "1. The following AI practices shall be prohibited:", "(a) the placing ...").
Every citable unit (article, paragraph, point, subpoint; annex, annex point) is a span of it,
and a unit's span contains its children's. Chunks are spans too, so whether a retrieved chunk
contains an expected passage is decided by span overlap, whatever the chunking.

Unit ids: art5, art5.1, art5.1.ba, art5.1.h.i, art3.12, art43.1.sp2.a (points of a later
subparagraph); anxIII, anxIII.1.a, and anxXI.s2.1 in annexes with sections. Labels are the
usual citations: "Article 5(1)(ba)", "Annex III, point 1(a)", "Annex XI, Section 2, point 1".

Amendments: the consolidated text marks each passage's source (▼B the original act, ▼M1 the
Digital Omnibus, Regulation (EU) 2026/1744). Each unit lists the acts its text comes from.
The preamble (recitals) is not part of a consolidated text.

python -m ai_act_qa.parse  -> data/processed/units.jsonl (+ canonical.txt)
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

from ai_act_qa.config import load_config, project_path

XHTML = "{http://www.w3.org/1999/xhtml}"
ORIGINAL = "32024R1689"
POINT_LABEL = re.compile(r"^\(?([0-9]+[a-z]*|[a-z]{1,3}|[ivxl]+)[.)]?$")
UNNAMED_ITEM = {"—", "-", "–", "•"}
ORDINALS = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}
TEXT_CLASSES = {"norm", "list", "normal", "tbl-norm", "norm inline-element", None, ""}


@dataclass
class Unit:
    id: str
    label: str
    kind: str  # article | paragraph | point | annex
    parent: str | None
    title: str  # the article's or annex's title
    chapter: str  # e.g. "Chapter III - High-risk AI systems, Section 2 - Requirements ..."
    start: int = -1
    end: int = -1
    sources: list[str] = field(default_factory=list)  # acts the text comes from (CELEX)


def tag(el: ET.Element) -> str:
    return el.tag.removeprefix(XHTML)


def raw_text(el: ET.Element) -> str:
    """Text of an element, with superscripts kept as powers ("10<sup>25</sup>" -> "10^25")."""
    inner = (el.text or "") + "".join(raw_text(c) + (c.tail or "") for c in el)
    return f"^{inner}" if el.get("class") == "superscript" else inner


def text_of(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", raw_text(el)).replace("\xa0", " ").strip()


def sentence_case(heading: str) -> str:
    """'CHAPTER II' -> 'Chapter II', 'PROHIBITED AI PRACTICES' -> 'Prohibited AI practices'."""
    words = [w if re.fullmatch(r"[IVXL]+|AI|EU|SMES?", w) else w.lower() for w in heading.split()]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def roman_or_point(label: str) -> str | None:
    """'(a) ' -> 'a', '1. ' -> '1', '(iv)' -> 'iv'; None for unnamed items (dashes)."""
    label = label.strip()
    if label in UNNAMED_ITEM or not label:
        return None
    match = POINT_LABEL.match(label)
    return match.group(1) if match else None


class Parser:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.pos = 0
        self.units: dict[str, Unit] = {}
        self.open: list[Unit] = []
        self.prefix = ""
        self.source = ORIGINAL
        self.chapter = ""
        self.section = ""
        self.annex_section = 0
        # Per article or paragraph: [current subparagraph, whether it has had points yet]. Text
        # after a list of points starts the next subparagraph, whose points are cited with it.
        self.subparagraph: dict[str, list] = {}

    # -- output ------------------------------------------------------------------------------
    def emit(self, text: str) -> None:
        text = (self.prefix + text).strip()
        self.prefix = ""
        if not text:
            return
        line = text + "\n"
        for unit in self.open:
            if unit.start < 0:
                unit.start = self.pos
            unit.end = self.pos + len(line)
            if self.source not in unit.sources:
                unit.sources.append(self.source)
        self.lines.append(line)
        self.pos += len(line)

    def push(self, unit: Unit) -> None:
        if unit.id in self.units:
            raise ValueError(f"duplicate unit id {unit.id} ({unit.label})")
        self.units[unit.id] = unit
        self.open.append(unit)

    def pop(self) -> None:
        unit = self.open.pop()
        if unit.start < 0:  # no text: not a citable unit
            del self.units[unit.id]

    # -- structure ---------------------------------------------------------------------------
    def chapter_path(self) -> str:
        return ", ".join(p for p in (self.chapter, self.section) if p)

    def walk(self, el: ET.Element) -> None:
        """Body level: chapters and sections give context; articles and annexes are units."""
        ident = el.get("id", "")
        if el.get("class") == "eli-subdivision" and ident.startswith("art_"):
            self.article(el)
            return
        if ident.startswith("anx_") and "." not in ident:
            self.annex(el)
            return
        if ident.startswith("cpt_"):
            self.heading(el, section=".sct_" in ident)
        for child in el:
            if child.get("class") in {"title-division-1", "title-division-2"}:
                continue
            self.walk(child)

    def heading(self, el: ET.Element, section: bool) -> None:
        parts = [text_of(c) for c in el if c.get("class") in {"title-division-1",
                                                               "title-division-2"}]  # fmt: skip
        name = " - ".join(sentence_case(p) if p.isupper() else p for p in parts)
        if section:
            self.section = name
        else:
            self.chapter, self.section = name, ""

    def article(self, el: ET.Element) -> None:
        number = el.get("id").removeprefix("art_")
        title_el = el.find(f"{XHTML}div[@class='eli-title']")
        title = text_of(title_el).rstrip("'’") if title_el is not None else ""  # one has a stray '
        self.push(Unit(f"art{number}", f"Article {number}", "article", None, title,
                       self.chapter_path()))  # fmt: skip
        self.emit(f"Article {number} - {title}" if title else f"Article {number}")
        for child in el:
            self.content(child)
        while self.open[-1].kind != "article":  # the last paragraph runs to the article's end
            self.pop()
        self.pop()

    def annex(self, el: ET.Element) -> None:
        number = el.get("id").removeprefix("anx_")
        titles = [text_of(c) for c in el if c.get("class") in {"title-annex-1", "title-annex-2"}
                  and not text_of(c).upper().startswith("ANNEX")]  # fmt: skip
        title = titles[0] if titles else ""
        self.annex_section = 0
        self.push(Unit(f"anx{number}", f"Annex {number}", "annex", None, title, ""))
        self.emit(f"Annex {number} - {title}" if title else f"Annex {number}")
        for child in el:
            if child.get("class") in {"title-annex-1", "title-annex-2"}:
                continue
            self.content(child)
        self.pop()

    # -- content -----------------------------------------------------------------------------
    def content(self, el: ET.Element) -> None:
        cls = el.get("class") or ""
        name = tag(el)
        if cls == "modref":
            link = el.find(f"{XHTML}a")
            title = link.get("title", "") if link is not None else ""
            self.source = title.split(":")[0].strip() or ORIGINAL  # a deletion leaves only this
            return
        if cls in {"title-article-norm", "eli-title", "footnote"} or name in {"hr", "br", "col"}:
            return
        if (
            name == "div"
            and cls == "norm"
            and el.find(f"{XHTML}span[@class='no-parag']") is not None
        ):
            self.paragraph(el)
        elif name == "div" and cls == "grid-container grid-list":
            self.point(el)
        elif name == "table":
            for row in el.iter(f"{XHTML}tr"):
                cells = [text_of(c) for c in row if tag(c) == "td"]
                if any(cells):
                    self.emit(" | ".join(c for c in cells if c))
        elif cls.startswith("title-gr-seq-level"):
            if cls.endswith("1"):
                self.annex_section += 1
            self.emit(text_of(el))
        elif name == "p" or (name == "div" and not any(tag(c) in {"div", "p", "table"}
                                                       for c in el)):  # fmt: skip
            if cls in TEXT_CLASSES or cls.startswith(("norm", "list", "title")):
                state = self.subparagraph.get(self.open[-1].id)
                if state and state[1]:
                    state[:] = [state[0] + 1, False]
                self.emit(text_of(el))
        else:
            for child in el:
                self.content(child)

    def paragraph(self, el: ET.Element) -> None:
        article = self.open[0]
        number = text_of(el.find(f"{XHTML}span[@class='no-parag']")).rstrip(".")
        if self.open[-1].kind == "point":  # quoted text of another act (amending provisions)
            self.prefix = f"{number}. "
            for child in el:
                if child.get("class") != "no-parag":
                    self.content(child)
            return
        if self.open[-1].kind == "paragraph":  # the previous paragraph ends here
            self.pop()
        self.push(Unit(f"{article.id}.{number}", f"{article.label}({number})", "paragraph",
                       article.id, article.title, article.chapter))  # fmt: skip
        self.prefix = f"{number}. "
        for child in el:
            if child.get("class") != "no-parag":
                self.content(child)
        # left open: unnumbered subparagraphs and lists that follow belong to it

    def point(self, el: ET.Element) -> None:
        label_el = el.find(f"{XHTML}div[@class='list grid-list-column-1']")
        body = el.find(f"{XHTML}div[@class='grid-list-column-2']")
        raw = text_of(label_el) if label_el is not None else ""
        name = roman_or_point(raw)
        if name is None or body is None:  # an unnamed list item: text of the enclosing unit
            self.prefix = f"{raw} " if raw else ""
            for child in body if body is not None else []:
                self.content(child)
            return
        parent = self.open[-1]
        state = self.subparagraph.setdefault(parent.id, [1, False])
        state[1] = True
        if parent.kind in {"article", "paragraph"} and state[0] > 1:
            uid = f"{parent.id}.sp{state[0]}.{name}"
            label = f"{parent.label}, {ORDINALS[state[0]]} subparagraph, point ({name})"
        elif parent.kind == "annex" and self.annex_section:
            section = self.annex_section
            uid = f"{parent.id}.s{section}.{name}"
            label = f"{parent.label}, Section {section}, point {name}"
        elif parent.kind == "annex":
            point = f"({name})" if name.isalpha() else name
            uid, label = f"{parent.id}.{name}", f"{parent.label}, point {point}"
        else:
            uid, label = f"{parent.id}.{name}", f"{parent.label}({name})"
        self.push(Unit(uid, label, "point", parent.id, parent.title, parent.chapter))
        self.prefix = f"{raw} "
        for child in body:
            self.content(child)
        self.pop()


def parse(xhtml: bytes) -> tuple[str, list[Unit]]:
    """-> (canonical text, units in document order)."""
    root = ET.fromstring(xhtml)
    body = root.find(f"{XHTML}body")
    parser = Parser()
    parser.walk(body)
    return "".join(parser.lines), list(parser.units.values())


def load_units() -> tuple[str, list[Unit]]:
    source = load_config()["source"]
    out = project_path(source["units_path"])
    canonical = (out.parent / "canonical.txt").read_text(encoding="utf-8")
    units = [Unit(**json.loads(line)) for line in out.read_text(encoding="utf-8").splitlines()]
    return canonical, units


def main() -> None:
    source = load_config()["source"]
    canonical, units = parse(project_path(source["path"]).read_bytes())
    out = project_path(source["units_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "canonical.txt").write_text(canonical, encoding="utf-8")
    out.write_text("".join(json.dumps(asdict(u), ensure_ascii=False) + "\n" for u in units),
                   encoding="utf-8")  # fmt: skip
    kinds = {k: sum(u.kind == k for u in units) for k in ("article", "paragraph", "point", "annex")}
    amended = sum(any(s != ORIGINAL for s in u.sources) for u in units)
    print(f"{len(units)} units {kinds}, {amended} with amended text; "
          f"{len(canonical.split()):,} words")  # fmt: skip


if __name__ == "__main__":
    main()
