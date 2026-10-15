"""Validate and run the frozen deterministic Reactor A control."""

from fractions import Fraction
import json
from pathlib import Path

from evaluator import evaluate_package
from validator import load_and_validate


def _render(value):
    if type(value) is Fraction:
        return str(value)
    if type(value) is tuple:
        return [_render(item) for item in value]
    return value


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    package = load_and_validate(directory)
    result = {key: _render(value) for key, value in evaluate_package(package).items()}
    result.update({
        "scope": package["control"]["scope"],
        "scientific_status_effect": package["control"]["scientific_status_effect"],
        "campaign_status": package["control"]["campaign_status"],
        "validated": True,
    })
    print(json.dumps(result, sort_keys=True, indent=2))
