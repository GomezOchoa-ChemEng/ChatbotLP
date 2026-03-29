import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.domain.sampat2019 import (
    CANONICAL_NOTATION,
    SECTION23_CONCEPTS,
    get_section_metadata,
    get_theorem_metadata,
    infer_benchmark_case,
)


def test_theorem_registry_lookup():
    theorem = get_theorem_metadata("theorem_1")
    assert theorem is not None
    assert theorem["target_section"] == "2.2"


def test_section_lookup():
    section = get_section_metadata("2.3")
    assert section is not None
    assert "negative bids" in " ".join(section["themes"])


def test_benchmark_case_inference():
    assert infer_benchmark_case(False, False) == "Case A"
    assert infer_benchmark_case(False, True) == "Case B"
    assert infer_benchmark_case(True, True) == "Case C"
    assert "negative_bid" in CANONICAL_NOTATION["economics"]
    assert "negative_prices" in SECTION23_CONCEPTS

