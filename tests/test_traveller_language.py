"""No harness vocabulary in the traveller's path.

The verdict used to print `schema_valid`, `transit_feasible` and `free_block`
in monospace next to someone's holiday, because `checkRow` rendered the check's
code name. Every check already carries a plain sentence; that is the label now,
and the snake_case name only appears with the developer view switched on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "wayfinder" / "web" / "index.html"


def markup() -> str:
    """What actually reaches a reader: text nodes, plus the attributes a
    person sees or a screen reader announces.

    Not element ids or class names — `id="o-interrupts"` is a handle for the
    code, not a word anyone reads. Jargon inside `<script>` is fine too:
    `check_itinerary` is the name of a real thing, and the point is only that
    it never gets rendered.
    """
    text = PAGE.read_text(encoding="utf-8")
    text = re.sub(r"<script.*?</script>", "", text, flags=re.S)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    spoken = " ".join(
        re.findall(r'(?:placeholder|title|aria-label|alt)="([^"]*)"', text)
    )
    return re.sub(r"<[^>]+>", " ", text) + " " + spoken


#: Words that mean something to whoever built this and nothing to a traveller.
JARGON = [
    "subagent", "harness", "schema", "repair loop", "tool call",
    "interrupt", "deep agent", "SKILL.md", "evaluator", "constraint checker",
]


@pytest.mark.parametrize("word", JARGON)
def test_the_page_does_not_say_it(word):
    assert word.lower() not in markup().lower(), f"{word!r} is visible to a traveller"


def test_no_tool_names_in_the_markup():
    """`finalize_itinerary` was a checkbox label. It is now "finishing the
    plan", and the mapping lives in the script where it belongs."""
    visible = markup()
    for tool in ("finalize_itinerary", "check_itinerary", "web_search",
                 "geocode", "estimate_travel", "fx_convert", "write_todos"):
        assert tool not in visible, f"{tool!r} is a label, not a name to show"


def test_no_snake_case_anywhere_visible():
    """A catch-all for the next one of these, since the specific list above
    only covers what exists today."""
    leaked = set(re.findall(r"\b[a-z]{3,}_[a-z_]{3,}\b", markup()))
    # `data-*` attributes and the like are markup, not prose.
    leaked -= {"data_mode", "aria_label"}
    assert not leaked, f"snake_case in the traveller's view: {sorted(leaked)}"


def test_every_check_has_a_sentence_to_show_instead_of_its_name():
    """The label comes from `CHECK_DESCRIPTIONS`, so a check without one would
    fall back to its own code name in the UI."""
    from wayfinder.verify import CHECK_DESCRIPTIONS, CheckResult

    for name in CHECK_DESCRIPTIONS:
        described = CheckResult(name, "soft").description
        assert described != name
        assert "_" not in described, f"{name} still reads like code"
        assert described[0].isupper(), f"{name}'s description is not a sentence"


def test_the_verdict_labels_are_written_for_a_person():
    page = PAGE.read_text(encoding="utf-8")
    for phrase in ("Everything checks out", "One thing needs fixing",
                   "What had to be true", "Worth knowing", "Your trip"):
        assert phrase in page
    for gone in ("Passed all ${hard.length} hard constraints",
                 "Quality warnings — advisory only",
                 "Harness &amp; approvals"):
        assert gone not in page, f"{gone!r} survived"


def test_the_code_name_is_kept_for_the_developer_view():
    """Removed from the traveller's path, not deleted — it is the thing you
    grep for when a check misbehaves."""
    page = PAGE.read_text(encoding="utf-8")
    assert 'classList.contains("dev")' in page
    assert 'el("div", "ds mono", c.name)' in page


def test_the_developer_view_is_off_until_asked_for():
    page = PAGE.read_text(encoding="utf-8")
    assert "#rawwrap { display: none; }" in page
    assert "body.dev #rawwrap { display: block; }" in page
    assert 'id="o-dev"' in page and "checked" not in page.split('id="o-dev"')[1][:40]
