"""The eval dataset: load it, validate it, push it to LangSmith."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langsmith import Client

from wayfinder.schema import TripSpec

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "dataset.yaml"
DEFAULT_DATASET_NAME = "wayfinder-trips"


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    spec: TripSpec

    @property
    def should_refuse(self) -> bool:
        return self.spec.should_refuse

    def inputs(self) -> dict[str, Any]:
        # `should_refuse` is stripped: it is the answer key, and handing it to
        # the agent would turn the hardest cases in the set into a giveaway.
        return {
            "name": self.name,
            "category": self.category,
            "spec": self.spec.model_dump(mode="json", exclude={"should_refuse"}),
        }

    def reference(self) -> dict[str, Any]:
        return {"should_refuse": self.should_refuse, "category": self.category}


def load_cases(path: Path | None = None) -> list[Case]:
    """Read and validate the dataset.

    Specs are parsed into `TripSpec` here so a malformed case fails at load
    time rather than twenty minutes into an eval run.
    """
    raw = yaml.safe_load((path or DATASET_PATH).read_text(encoding="utf-8"))
    cases = []
    for entry in raw:
        spec_data = dict(entry["spec"])
        spec_data["should_refuse"] = entry.get("should_refuse", False)
        cases.append(
            Case(
                name=entry["name"],
                category=entry.get("category", "uncategorised"),
                spec=TripSpec.model_validate(spec_data),
            )
        )

    names = [c.name for c in cases]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        msg = f"duplicate case names in the dataset: {sorted(duplicates)}"
        raise ValueError(msg)
    return cases


def case_by_name(name: str, path: Path | None = None) -> Case:
    for case in load_cases(path):
        if case.name == name:
            return case
    msg = f"no case named {name!r}"
    raise KeyError(msg)


def sync_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    path: Path | None = None,
    client: Client | None = None,
) -> str:
    """Create or refresh the LangSmith dataset from `evals/dataset.yaml`.

    Examples are replaced wholesale rather than merged. Merging would leave
    stale cases behind after an edit, and a dataset that quietly disagrees with
    the file in the repo makes every experiment run against it unreproducible.
    """
    client = client or Client()
    cases = load_cases(path)

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        stale = list(client.list_examples(dataset_id=dataset.id))
        if stale:
            client.delete_examples(example_ids=[e.id for e in stale])
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Trip specs spanning easy, tight-budget, constraint-dense, "
                "shoulder-season and deliberately infeasible cases."
            ),
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": case.inputs(),
                "outputs": case.reference(),
                # The category is also a LangSmith *split*, not just a label.
                # Splits are filterable in the UI and selectable when running
                # an experiment, so "how does this arm do on the infeasible
                # cases" becomes a question you can ask directly — and those
                # subsets behave so differently that one average over all
                # twenty hides more than it shows.
                #
                # `split` is a first-class field, not a metadata key. Writing
                # `metadata["dataset_split"]` — which is what the API *reads
                # back* — is silently ignored on create, and every example
                # stays in "base".
                "split": [case.category],
                "metadata": {"category": case.category, "name": case.name},
            }
            for case in cases
        ],
    )
    return dataset_name


def splits(path: Path | None = None) -> dict[str, list[str]]:
    """Case names grouped by split, for `--split` and for the console."""
    grouped: dict[str, list[str]] = {}
    for case in load_cases(path):
        grouped.setdefault(case.category, []).append(case.name)
    return grouped
