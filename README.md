# Wayfinder

A constraint-solving trip planner built on [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview), observed with [LangSmith](https://www.langchain.com/langsmith-platform).

Give it a city, dates, a budget and some constraints. It researches, plans, and
produces an itinerary that has been **checked in code** — budget, opening hours,
travel times, walking distance, downtime — not just vibe-checked by another
model.

It never books anything. It produces a plan and the links behind it.

## Why it's built this way

The output is a validated JSON document, not prose. That single decision is what
makes the project worth building:

- **The evaluator is the spec.** `wayfinder/verify.py` was written before the
  agent existed. It decides, in pure Python, whether a plan is any good.
- **The agent optimises against the thing that grades it.** The same
  `check_payload` function is exposed to the agent mid-run as the
  `check_itinerary` tool *and* called by the LangSmith evaluators at scoring
  time. They cannot drift apart.
- **Judges are the exception, not the rule.** Ten code evaluators; three LLM
  judges, for the things code genuinely can't reach — preference fit,
  groundedness, readability.

```
research (subagents) → draft itinerary.json → check → repair → check → render
```

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys
```

You need `OPENROUTER_API_KEY` and `TAVILY_API_KEY` to plan, plus
`LANGSMITH_API_KEY` to run experiments. `WAYFINDER_USER_AGENT` is required by
Nominatim's usage policy — set it to something that identifies you.
`OPENAI_API_KEY` unlocks the spoken intake; without it the typed paths still
work. `ANTHROPIC_API_KEY` is only needed if you pass `--model anthropic:…`.

Models are named `provider:model`. `openrouter:` specs are routed through
OpenRouter — the default planner is `openrouter:deepseek/deepseek-v4-flash` at
$0.14/$0.28 per MTok, which is what makes sweeping the experiment matrix
affordable. Everything else falls through to LangChain's own resolution, so
`anthropic:claude-sonnet-5` still works unchanged.

## Use

```bash
uv run wayfinder plan trips/lisbon-oct.yaml
```

Writes a run directory containing everything the agent did:

```
runs/lisbon-2026-10-12-20260803T1422/
  itinerary.json      # the checked artifact
  itinerary.md        # the readable rendering
  constraints.json    # the checker's verdict
  sources.md          # every claim traced to a URL
  research/           # what the subagents found
  spec.yaml           # the input, snapshotted
  config.json         # which harness config produced this
  skills/             # the skills as they were at run time
```

Re-check a plan at any time — the same code path the agent and the evaluators use:

```bash
uv run wayfinder check runs/lisbon-2026-10-12-20260803T1422/
```

## The web UI

```bash
uv run wayfinder serve
```

Three ways to describe a trip, all writing into the same `TripSpec`:

- **Form** — structured fields.
- **Questions** — a 13-step wizard, client-side and deterministic.
- **🎙 Ramble** — a spoken conversation. Hit *Start conversation* and a voice
  interviewer asks about the trip until it has enough to plan, one question at
  a time, filling the board as you answer. Corrections win: *"four hundred
  euros… no, make it six hundred"* lands as €600. Plan stays disabled until
  destination, dates and budget are all in.

  It runs over WebRTC against OpenAI's Realtime API. The browser never gets the
  API key — the server mints a short-lived client secret from it. Falls back to
  browser dictation (Web Speech API) or a pasted monologue, both of which
  re-extract the whole transcript on every pause.

  GPT-Live-1 and GPT-Live-1 mini are not in the API yet: they shipped 8 July
  2026 as the ChatGPT voice experience with a developer waitlist and no
  announced GA. The model is `REALTIME_MODEL`, so switching is one line.

While it runs you get the **live feed** (every tool call, subagent dispatch and
constraint check) and a **map that fills in as the agents scout** — each
successful geocode drops a dot. When it finishes, the route renders: numbered
pins, one colour per day, clickable both ways with the **journey cards** below.

**Human-in-the-loop**: tick any tool under *Ask me before…* and the run pauses
for approve / edit / send-back. `finalize_itinerary` is the default gate — the
agent's last action, so you see the finished plan before it's called done.

**⇩ Export PDF** compiles a paginated document: per-day route tables with the
transit legs between stops, ratings, per-day cost/walking/transit totals, and
the full source bibliography.

**Past trips** are read from `runs/` on disk, so a restart doesn't orphan them.

### Ratings

The agent checks Google-review scores (`venue_rating`) for sights and
restaurants and uses them as one selection criterion among several — a
400-review neighbourhood tasca can still beat a 20,000-review tourist machine.
Scores appear on the cards, in map popups, in `itinerary.md` and in the PDF.

These come from review data quoted in search results, not the billed Places
API, so **every score carries a source URL** and an unfound rating is left
unset rather than guessed.

## Evaluation

```bash
uv run wayfinder sync-dataset                 # push evals/dataset.yaml
uv run wayfinder eval --arm baseline          # one experiment
uv run wayfinder eval --arm baseline --repetitions 3   # measure the noise floor
uv run wayfinder matrix                       # every arm, then compare in the UI
uv run wayfinder eval-local --case lisbon-easy         # no LangSmith needed
```

The dataset spans easy, tight-budget, constraint-dense, shoulder-season and
**deliberately infeasible** cases. On the last group the correct answer is
`feasible: false` with the conflict named — an agent that fabricates a plan
instead scores zero on `correctly_refused` while looking healthy everywhere
else. That's the case that exposes a bad agent.

### The arms

Each differs from the baseline in exactly one flag, so a column reads as "what
did this component buy us?"

| Arm | Question |
|---|---|
| `no-repair-loop` | Does an in-loop verifier actually improve the pass rate? |
| `no-skills` | Do the `SKILL.md` files earn their tokens? |
| `no-subagents` | Is context isolation worth the dispatches? |
| `one-researcher` | Three specialists, or one generalist? |
| `cheap-subagents` | Does a cheap model do the research just as well? |
| `opus-throughout` | Is Opus 5 worth roughly double the spend over the Sonnet 5 baseline? |

**Run `--repetitions 3` before you believe any delta.** A 5% improvement means
nothing until you know how much re-running the same config moves the number.

## Costs

A baseline sweep is ~20 agentic trips on Sonnet 5 ($3/$15 per MTok, currently
$2/$10 introductory through 2026-08-31). The `opus-throughout` arm runs the
same sweep on Opus 5 at $5/$25. Iterate on a
few cases first (`eval-local --case …`), then go wide. Every external call is
cached to `.cache/`, so re-runs are cheap and — more importantly — reproducible:
a delta between two experiments comes from the thing you varied, not from the
web changing underneath you.

## Tests

```bash
uv run pytest -q
```

The suite is offline. It covers the checker (including a fixture that trips
every single check — add a check without seeding a failure and a test tells
you), the agent plumbing, the filesystem guardrail, and every evaluator.
