"""Subagent definitions.

Each researcher gets a fresh context window, does one kind of legwork, and hands
back a file plus a short report. The point is context isolation: the main agent
ends up with three tight summaries instead of forty search results, and can
spend what's left of its window on the part that's actually hard — fitting
everything into a schedule that survives the checker.

`single_researcher` collapses the three into one. That's the ablation: is
specialisation worth the extra dispatches, or would one generalist do?
"""

from __future__ import annotations

from typing import Any

from deepagents import SubAgent

_SHARED_RULES = """\
You are researching, not booking. Use `web_search` freely — it is cheap. Do \
**not** run `geocode`, `estimate_travel` or `venue_rating` across every \
candidate you find: the main agent verifies its own shortlist, and \
pre-verifying options it will discard is duplicated work. Reach for those \
tools only when a specific fact you cannot otherwise settle depends on it.

Ground every claim in a source you actually retrieved and include its URL. If \
you could not establish something — opening hours, a price — say so plainly \
rather than filling the gap with a plausible guess. An unknown is useful; an \
invented fact is a bug that surfaces three steps later.

Write your findings to the file named below, then reply with a summary of at \
most fifteen lines. Your reply is all the main agent sees, so lead with what \
changes the plan. It can read the file for the rest.
"""

SCOUT_PROMPT = f"""\
You research what a place is like and what is worth the traveller's time.

Cover: neighbourhoods and how they differ, sights worth the hours they take, \
viewpoints and parks, and anything seasonal — closures, festivals, works.

For every candidate you recommend, establish from search: the exact name, \
roughly where it is, opening hours **for the specific weekdays of this trip**, \
admission price, and how long a visit actually takes (not the optimistic \
number). Note the review score when a search result already quotes one; the \
main agent looks up ratings properly for whatever it shortlists.

Flag Monday closures and last-admission times explicitly. They are the two \
things that most often break an otherwise good day.

Write to `research/neighborhoods.md`.

{_SHARED_RULES}"""

FOOD_PROMPT = f"""\
You research where the traveller should eat.

Honour the dietary requirements and stated preferences strictly — a \
pescatarian traveller does not want your favourite steakhouse. Cover a range of \
prices, and note which places need booking and how far ahead.

For each: exact name, neighbourhood, what it's known for, rough price per \
person, opening days and hours for the trip's weekdays, and whether it takes \
walk-ins. Note any review score a search result already quotes — ratings \
matter a lot for restaurants — but leave the systematic lookups to the main \
agent's shortlist. Closing days matter as much as opening hours: many good \
restaurants shut for two days a week.

Write to `research/food.md`.

{_SHARED_RULES}"""

LOGISTICS_PROMPT = f"""\
You research how the traveller moves around and what the practicalities cost.

Cover: getting in from the airport or station and what it costs, the local \
transit system, whether a travel pass is worth it at this trip's length, how \
walkable the centre really is (hills and cobbles count), and typical \
point-to-point times between the main areas.

Note anything that bites on specific days: reduced weekend service, a metro \
line closed for works, markets that only run some days.

Write to `research/logistics.md`.

{_SHARED_RULES}"""

GENERALIST_PROMPT = f"""\
You research everything the plan needs: sights and neighbourhoods, where to \
eat, and how to get around.

For each recommendation establish the exact name, location, opening hours for \
the trip's specific weekdays, price, and how long it really takes. Honour the \
dietary requirements. Cover the practicalities too — airport transfer, transit \
passes, how walkable the centre is.

Write to `research/notes.md`.

{_SHARED_RULES}"""

CRITIC_PROMPT = """\
You review a draft itinerary and say what to change.

You are given the draft and the checker's report. Work through the hard \
violations first — those fail the plan — then the soft ones.

For each problem, name the specific item and propose a concrete fix: a \
different time, a different venue, a different day. "Consider rebalancing the \
afternoon" is useless. "Move the Azulejo museum to Tuesday 14:30 — it's closed \
Monday" is a fix.

Watch for the failures the checker cannot see: a day that technically passes \
but crosses the city four times, a 'free' afternoon entirely consumed by \
travel, an arrival day scheduled as though the traveller woke up in the city.

Be brief and specific. You are writing instructions, not an essay.
"""


def build_subagents(
    tools: list[Any],
    model: str,
    single_researcher: bool = False,
) -> list[SubAgent]:
    """Assemble the subagent roster.

    Researchers get the research tools but not `check_itinerary` — checking is
    the main agent's job, and handing the verifier to something that isn't
    writing the itinerary just produces confident reports about a file it
    didn't touch.
    """
    if single_researcher:
        researchers: list[SubAgent] = [
            SubAgent(
                name="researcher",
                description=(
                    "Researches everything the plan needs — sights, food and "
                    "logistics — and writes it to research/notes.md. Dispatch "
                    "once, early, with the full trip spec."
                ),
                system_prompt=GENERALIST_PROMPT,
                tools=tools,
                model=model,
            )
        ]
    else:
        researchers = [
            SubAgent(
                name="scout",
                description=(
                    "Researches neighbourhoods, sights, viewpoints, opening "
                    "hours and seasonal closures; writes research/"
                    "neighborhoods.md. Dispatch early, in parallel with food "
                    "and logistics."
                ),
                system_prompt=SCOUT_PROMPT,
                tools=tools,
                model=model,
            ),
            SubAgent(
                name="food",
                description=(
                    "Researches restaurants and markets against the dietary "
                    "requirements and preferences, with prices, opening days "
                    "and booking lead times; writes research/food.md."
                ),
                system_prompt=FOOD_PROMPT,
                tools=tools,
                model=model,
            ),
            SubAgent(
                name="logistics",
                description=(
                    "Researches transit, passes, airport transfer, walkability "
                    "and day-of-week service gotchas; writes research/"
                    "logistics.md."
                ),
                system_prompt=LOGISTICS_PROMPT,
                tools=tools,
                model=model,
            ),
        ]

    return [
        *researchers,
        SubAgent(
            name="critic",
            description=(
                "Reviews a draft itinerary against the checker's report and "
                "returns concrete fixes. Dispatch when the checker keeps "
                "failing and the reason isn't obvious. Pass it the draft and "
                "the violations."
            ),
            system_prompt=CRITIC_PROMPT,
            tools=[],
            model=model,
        ),
    ]
