---
name: itinerary-format
description: The exact JSON contract for itinerary.json — field-by-field reference, a worked example, and the mistakes the checker catches most often. Read this before writing or repairing an itinerary.
---

# Writing `itinerary.json`

The system prompt has the shape. This is the detail: what each field means, and
where plans usually go wrong.

## Field reference

### Top level

| Field | Notes |
|---|---|
| `destination` | Free text; echo the spec's destination. |
| `currency` | ISO 4217. **Must equal the budget's currency** — the checker will not convert for you. |
| `feasible` | `true` unless the spec genuinely cannot be satisfied. |
| `infeasibility_reason` | Required when `feasible` is `false`, ignored otherwise. Name the colliding constraints. |
| `notes` | Two or three sentences on the shape of the trip. Not a sales pitch. |
| `days` | One object per date in the range. In order. |

### `days[]`

| Field | Notes |
|---|---|
| `date` | `YYYY-MM-DD`. Every date in the spec range, no more, no fewer. |
| `summary` | One line. Optional. |
| `items` | Chronological, non-overlapping. |

### `items[]`

| Field | Notes |
|---|---|
| `start` / `end` | `HH:MM`, 24-hour, local. `end` must be strictly after `start`. |
| `kind` | `activity` \| `meal` \| `transit` \| `rest` \| `travel`. Only `activity` counts toward the pace ceiling. |
| `meal_slot` | Required when `kind` is `meal`: `breakfast` \| `lunch` \| `dinner`. |
| `title` | What the traveller is doing, e.g. "Dinner at Cervejaria Ramiro". |
| `estimated_cost` | Number, **whole party**, budget currency. Free things are `0`, not omitted. |
| `sources` | URLs backing the hours and prices you claimed. |
| `venue` | Omit for items with no location (a rest block). |
| `transit_from_previous` | How the traveller got here from the previous item. |
| `note` | Anything the traveller needs to know — booking lead time, which entrance. |

### `venue`

```json
{
  "name": "Museu Nacional do Azulejo",
  "address": "R. Me. Deus 4, 1900-312 Lisboa",
  "lat": 38.7247,
  "lon": -9.1136,
  "opening_hours": {
    "week": {
      "mon": [{"open": "10:00", "close": "18:00"}],
      "tue": []
    }
  }
}
```

**The `week` map has three states, and the difference matters.**

| State | Means | Consequence |
|---|---|---|
| Weekday **absent** | You never established the hours | Soft warning — go and look |
| Weekday present, **empty list** | Closed that day | Hard failure if you schedule anything |
| Weekday present, **non-empty** | Open in those windows | Hard failure if your slot falls outside them |

Never guess. An absent weekday is an honest "unknown"; inventing `09:00-18:00`
is a claim the groundedness check will hold you to.

Use two ranges for a lunch closure:
`"tue": [{"open": "10:00", "close": "13:00"}, {"open": "15:00", "close": "19:00"}]`

### `transit_from_previous`

```json
{"mode": "walk", "minutes": 20, "distance_km": 1.4, "note": "uphill"}
```

`mode` is `walk` \| `bike` \| `transit` \| `taxi` \| `other`. Take `minutes` and
`distance_km` from `estimate_travel` rather than estimating by eye.
`distance_km` on a `walk` leg is what the daily walking cap is measured from —
omit it and a mobility limit silently can't be enforced.

## Worked example

```json
{
  "destination": "Lisbon, Portugal",
  "currency": "EUR",
  "feasible": true,
  "notes": "Two unhurried days: the eastern viewpoints, then Belem.",
  "days": [
    {
      "date": "2026-10-12",
      "summary": "Viewpoints in the east, dinner at the market.",
      "items": [
        {
          "start": "10:30", "end": "12:00", "kind": "activity",
          "title": "Miradouro da Senhora do Monte",
          "estimated_cost": 0,
          "sources": ["https://www.visitlisboa.com/miradouro-senhora-do-monte"],
          "venue": {
            "name": "Miradouro da Senhora do Monte",
            "lat": 38.7186, "lon": -9.1329,
            "opening_hours": {"week": {"mon": [{"open": "00:00", "close": "23:59"}]}}
          }
        },
        {
          "start": "14:30", "end": "16:00", "kind": "activity",
          "title": "Museu Nacional do Azulejo",
          "estimated_cost": 20,
          "sources": ["https://www.museudoazulejo.gov.pt/"],
          "transit_from_previous": {"mode": "walk", "minutes": 20, "distance_km": 1.4},
          "venue": {
            "name": "Museu Nacional do Azulejo",
            "lat": 38.7247, "lon": -9.1136,
            "opening_hours": {"week": {"mon": [{"open": "10:00", "close": "18:00"}]}}
          }
        },
        {
          "start": "19:30", "end": "21:00", "kind": "meal", "meal_slot": "dinner",
          "title": "Dinner at Time Out Market",
          "estimated_cost": 55,
          "sources": ["https://www.timeoutmarket.com/lisboa/"],
          "transit_from_previous": {"mode": "transit", "minutes": 25},
          "venue": {
            "name": "Time Out Market",
            "lat": 38.7071, "lon": -9.1459,
            "opening_hours": {"week": {"mon": [{"open": "10:00", "close": "23:59"}]}}
          }
        }
      ]
    }
  ]
}
```

## What the checker catches most often

1. **A missing transit leg between two different venues.** Hard failure. Leaving
   it out is how an impossible day looks possible, so it's treated as one.
2. **Extra fields.** Unknown keys are rejected outright. No `"weather"`, no
   `"tips"`, no `"day_number"`.
3. **Costs in the wrong currency.** Convert with `fx_convert` first.
4. **A date missing from `days`**, or one that isn't in the trip range.
5. **Overlapping items** — scheduling two things at once to make a day fit.
6. **`kind: "meal"` without `meal_slot`.** Schema error, rejects the whole file.
7. **`23:59` vs `24:00`.** There is no 24:00. Late-night closing is `23:59`.
