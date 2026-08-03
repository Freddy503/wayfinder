---
name: transit-realism
description: How to sequence a day so it survives contact with a real city — transit legs, buffers, opening hours, and the geography mistakes that make a plan look fine on paper and fail on the ground. Read before assigning times to items.
---

# Making a day actually work

Most failed itineraries fail here. Not because a venue was wrong, but because
the day assumed the traveller could teleport.

## Sequence before you schedule

1. **Group by geography first.** Cluster the day's candidates into one
   neighbourhood, or two adjacent ones. A day that crosses the city three times
   loses an hour each way and nobody enjoys it.
2. **Geocode everything.** `geocode` each venue. Coordinates are what make
   travel times computable.
3. **Estimate every hop.** `estimate_travel` between consecutive stops. Do this
   *before* choosing times, not to justify times you already chose.
4. **Then place the times**, working around fixed points — a restaurant
   booking, a timed-entry ticket, a museum that shuts at 17:30.

## Buffers

`estimate_travel` returns a floor: straight-line distance with a routing factor.
It doesn't know about hills, closed streets, or a queue at the ticket desk.

- Add **10 minutes** on top of any estimate under 20 minutes.
- Add **15–20 minutes** on longer hops or anything involving a change.
- Add **20–30 minutes** before a timed entry or a booked table. Arriving early
  is free; arriving late loses the booking.

Record the *estimate* in `transit_from_previous.minutes` and let the buffer live
in the gap between items. The checker compares the leg against the gap, so a
buffer shows up as slack rather than as an inflated travel time.

## The rules the checker enforces, and why

**Every move between different venues needs a leg.** Not a courtesy — omitting
it is exactly how an impossible day passes review, so a missing leg is a hard
failure regardless of how comfortable the gap looks.

**The leg must fit the gap.** 40 minutes of travel into a 30-minute window is a
plan that breaks at 11am on day one.

**`max_transit_minutes` caps a single hop.** A traveller who said "nothing over
35 minutes" meant it. If two things you want are 50 minutes apart, they belong
on different days — not on the same day with a 50-minute leg you hoped nobody
would read.

**Downtime is net of travel.** `min_free_block_minutes` is measured between
scheduled items *minus* the travel into the next one. A three-hour gap
containing a 170-minute journey is ten minutes off, and the check reads it that
way.

## Opening hours

Look them up, per weekday, for the actual dates of the trip. The three states in
`opening_hours.week` — absent, empty, populated — mean unknown, closed, and open
respectively, and they are not interchangeable.

Traps worth checking every time:

- **Monday.** A large share of European museums close on Mondays.
- **Last admission**, typically 30–60 minutes before closing. If a museum closes
  at 18:00, a 17:45 arrival is a locked door. Schedule the *visit*, and make it
  end before closing.
- **Lunch closures.** Common in Iberia and Italy: two ranges, not one.
- **Seasonal hours.** Summer and winter timetables differ, and shoulder season
  is where the plan quietly breaks.
- **Market days.** Some markets run two days a week and are an empty square on
  the other five.

## Pacing a day

- **Relaxed**: 2–3 activities. **Standard**: 3–4. **Packed**: 5+, and even then
  only if they're close together.
- Meals are not activities, but they are 60–90 minutes each and they anchor the
  day.
- Big sights take longer than their listed duration. A major museum is 2–3
  hours, not the 90 minutes you'd like it to be.
- Leave the late afternoon loose. It's when plans slip, and it's the part of a
  trip people remember.

## Arrival and departure days

They are not full days. An arrival day starts when the traveller actually gets
in, plus an hour to drop bags — schedule one light thing, not three. A departure
day ends well before the flight. If the spec doesn't say when they land, say
what you assumed in the day's `summary`.
