---
name: budget-discipline
description: How to estimate trip costs honestly and keep a plan inside its budget — what to count, what a realistic price looks like, and how to cut when the total runs over. Read before pricing an itinerary or repairing a budget violation.
---

# Costing a trip

The budget is a hard constraint. An itinerary over the cap fails, and an
itinerary that fits only because things were left unpriced fails the traveller,
which is worse.

## What counts

Everything with a price tag that the traveller will pay **on the days you are
planning**, for the **whole party**:

- Admission and tickets
- Meals you schedule — including the ones you don't name a restaurant for
- Local transit: passes, metro fares, the airport transfer, taxis
- Tours, rentals, tastings

Check the spec's `budget.excludes`. `excludes: [flights]` means flights are the
traveller's problem, not that everything else is free. Accommodation is only
excluded if it says so.

`estimated_cost` is **per item, whole party**. Two adults at a €12 museum is
`24`, not `12`. This is the single most common costing error.

## Getting prices right

- Search for the current price rather than recalling one. Admission changes
  yearly; "about €10" for a major museum is usually wrong by 2026.
- Use the official site. Aggregators quote resale prices with a markup.
- Watch for concessions and combined tickets — a €25 combined ticket often
  beats two separate €16 ones.
- Free days exist. Many European municipal museums are free on the first Sunday
  of the month; if one lands in the trip, schedule around it.

**Meals.** Guess deliberately rather than vaguely. Per person, per meal, in a
mid-range European city:

| | Cheap | Mid | Notable |
|---|---|---|---|
| Breakfast | 4 | 8 | 15 |
| Lunch | 10 | 18 | 35 |
| Dinner | 15 | 30 | 65+ |

Adjust for the city — Lisbon and Porto sit below this, Copenhagen and Zurich
well above. Say which end you assumed in the item's `note`.

## When the total runs over

Cut in this order. The first two cost the traveller almost nothing.

1. **Re-check the arithmetic.** Party size double-counted, a per-person price
   entered as a party price, a currency left unconverted.
2. **Substitute, don't delete.** A viewpoint instead of a paid tower. A
   neighbourhood tasca instead of the guidebook restaurant. Most trips have one
   expensive item doing no more work than a free one.
3. **Downgrade one meal a day**, not all of them. One memorable dinner beats
   four adequate ones.
4. **Swap transit for walking** where the distance is short and the walking cap
   allows — but re-run `estimate_travel`, because the minutes change and the
   transit check will catch it if you don't.
5. **Drop a paid activity.** Last, and never a `must_do`.

If it still doesn't fit after all five, the spec is the problem. Say so:
`feasible: false` with the arithmetic — what the trip costs at its cheapest,
and what the budget allows.

## Leave headroom

Aim to land 10–15% under the cap. Estimates drift, and a plan that fits with
€3 to spare fits on paper only.

## Don't

- Don't set `estimated_cost` to `0` to make the budget work. Free means free.
- Don't quietly drop a `must_do` to save money — it fails a different check and
  ignores the one thing the traveller actually asked for.
- Don't pad prices "to be safe" either. An overstated total triggers cuts that
  weren't necessary.
