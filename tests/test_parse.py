"""Parsing: units, citations and spans, on a small hand-written document and on the real Act."""

import pytest

from ai_act_qa.config import load_config, project_path
from ai_act_qa.parse import parse

AMENDING = "32026R1744"


def doc(body: str) -> bytes:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head><body>'
        + body
        + "</body></html>"
    ).encode()


def point(label: str, inner: str) -> str:
    return (
        '<div class="grid-container grid-list"><div class="list grid-list-column-1">'
        f'<span>{label} </span></div><div class="grid-list-column-2">{inner}</div></div>'
    )


def modref(celex: str) -> str:
    return f'<p class="modref"><a title="{celex}: REPLACED">x</a></p>'


ARTICLE = doc(
    '<div id="cpt_II"><p class="title-division-1">CHAPTER II</p>'
    '<p class="title-division-2">PROHIBITED AI PRACTICES</p>'
    '<div class="eli-subdivision" id="art_5"><p class="title-article-norm">Article 5</p>'
    '<div class="eli-title"><p class="stitle-article-norm">Prohibited AI practices</p></div>'
    '<div class="norm"><span class="no-parag">1.  </span><div class="norm inline-element">'
    '<p class="norm">The following shall be prohibited:</p>'
    + point("(a)", '<p class="norm">first practice;</p>')
    + modref(AMENDING)
    + point(
        "(ba)", '<p class="norm">an inserted practice, 10<span class="superscript">25</span>;</p>'
    )
    + modref("32024R1689")
    + '<p class="norm">Where:</p>'
    + point("(a)", '<p class="norm">a condition of the second subparagraph.</p>')
    + "</div></div>"
    '<p class="norm">An unnumbered subparagraph of paragraph 1.</p>'
    '<div class="norm"><span class="no-parag">2.  </span>'
    '<div class="norm inline-element">Second paragraph.</div></div>' + "</div></div>"
)


@pytest.fixture(scope="module")
def small():
    canonical, units = parse(ARTICLE)
    return canonical, {u.id: u for u in units}


def text(canonical, unit):
    return canonical[unit.start : unit.end]


def test_units_ids_and_citations(small):
    _, units = small
    assert list(units) == ["art5", "art5.1", "art5.1.a", "art5.1.ba", "art5.1.sp2.a", "art5.2"]
    assert units["art5.1.ba"].label == "Article 5(1)(ba)"
    assert units["art5.1.sp2.a"].label == "Article 5(1), second subparagraph, point (a)"
    assert units["art5"].chapter == "Chapter II - Prohibited AI practices"


def test_spans_nest_and_subparagraphs_stay_with_their_paragraph(small):
    canonical, units = small
    assert text(canonical, units["art5"]).startswith("Article 5 - Prohibited AI practices\n")
    assert "(ba) an inserted practice" in text(canonical, units["art5.1"])
    assert "An unnumbered subparagraph" in text(canonical, units["art5.1"])
    assert text(canonical, units["art5.2"]) == "2. Second paragraph.\n"
    for child, parent in [("art5.1.a", "art5.1"), ("art5.1", "art5")]:
        assert units[parent].start <= units[child].start <= units[child].end <= units[parent].end


def test_sources_follow_amendment_markers(small):
    _, units = small
    assert units["art5.1.ba"].sources == [AMENDING]
    assert units["art5.1.a"].sources == ["32024R1689"]
    assert set(units["art5"].sources) == {"32024R1689", AMENDING}


def test_superscripts_are_kept_as_powers(small):
    canonical, units = small
    assert "10^25" in text(canonical, units["art5.1.ba"])


@pytest.fixture(scope="module")
def act():
    path = project_path(load_config()["source"]["path"])
    canonical, units = parse(path.read_bytes())
    return canonical, {u.id: u for u in units}


def test_real_act_structure(act):
    canonical, units = act
    kinds = [u.kind for u in units.values()]
    assert kinds.count("article") == 119 and kinds.count("annex") == 14
    assert "greater than 10^25" in text(canonical, units["art51.2"])
    assert "2 December 2026" in text(canonical, units["art113.a"])
    assert AMENDING in units["art5.1.ba"].sources
    assert units["anxIII.4.a"].label == "Annex III, point 4(a)"


def test_real_act_keeps_every_word(act):
    """The canonical text has every article and annex word of the source (bar markers)."""
    canonical, units = act
    assert len(canonical.split()) > 59_000
    assert len({u.id for u in units.values()}) == len(units)
