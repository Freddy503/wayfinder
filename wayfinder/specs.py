"""Loading trip specs and itineraries off disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from wayfinder.schema import Itinerary, TripSpec


def load_spec(path: str | Path) -> TripSpec:
    """Read a YAML (or JSON) trip spec.

    Validation errors surface here rather than three layers into an agent run,
    which is where you want a typo in a date to be caught.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TripSpec.model_validate(raw)


def load_itinerary_payload(path: str | Path) -> Any:
    """Read an itinerary as raw JSON, *unvalidated*.

    Deliberately not parsed into an `Itinerary`: a malformed document is a
    finding for `check_payload` to report, not an exception for the caller to
    catch.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_itinerary(path: str | Path) -> Itinerary:
    return Itinerary.model_validate(load_itinerary_payload(path))


def dump_itinerary(itinerary: Itinerary, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(itinerary.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
