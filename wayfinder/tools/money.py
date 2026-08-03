"""Currency conversion against a hand-maintained rate table.

Static rates are the right call here: they're deterministic, they need no API
key, and a few percent of drift is far below the noise floor of "a museum
ticket costs about 12". What matters is that a budget in one currency and a
price quoted in another end up comparable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_RATES_PATH = Path(__file__).with_name("rates.json")


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(_RATES_PATH.read_text(encoding="utf-8"))


def fx_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies.

    Use this whenever a price you found is quoted in a different currency from
    the trip's budget. Every `estimated_cost` in the itinerary must be in the
    budget's currency — the checker fails an itinerary priced in anything else
    rather than trying to guess.

    Args:
        amount: The amount to convert.
        from_currency: ISO 4217 code the amount is currently in, e.g. "USD".
        to_currency: ISO 4217 code to convert to, e.g. "EUR".

    Returns:
        `{"ok": true, "amount", "currency", "rate", "as_of"}`, or
        `{"ok": false, "reason": ...}` for an unknown currency.
    """
    table = _table()
    rates = table["rates"]
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()

    missing = [c for c in (src, dst) if c not in rates]
    if missing:
        return {
            "ok": False,
            "reason": f"no rate for {', '.join(missing)}",
            "known": sorted(rates),
        }

    # Rates are quoted per unit of the base currency, so cross-rate via base.
    converted = amount / rates[src] * rates[dst]
    return {
        "ok": True,
        "amount": round(converted, 2),
        "currency": dst,
        "rate": round(rates[dst] / rates[src], 6),
        "as_of": table["as_of"],
    }
