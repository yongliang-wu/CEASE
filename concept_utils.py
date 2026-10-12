"""Shared concept-list loading and validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path


def normalize_concept(value: str) -> str:
    concept = value.strip()
    if not concept:
        raise ValueError("Concept names must not be empty")
    return concept


def load_concepts(
    csv_path: str,
    *,
    column: str = "concept",
    start: int = 0,
    limit: int | None = None,
) -> list[str]:
    """Load an ordered, de-duplicated concept list with CSV quoting intact."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} must contain a {column!r} column")

        concepts: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            raw = row.get(column)
            if raw is None:
                raise ValueError(f"Missing {column!r} at {path}:{row_number}")
            concept = normalize_concept(raw)
            if concept not in seen:
                seen.add(concept)
                concepts.append(concept)

    if start < 0:
        raise ValueError("start must be non-negative")
    selected = concepts[start:] if limit is None else concepts[start : start + limit]
    if limit is not None and len(selected) != limit:
        raise ValueError(
            f"Expected {limit} concepts from {path} at offset {start}, found {len(selected)}"
        )
    return selected


def split_concepts(csv_path: str, *, per_split: int = 100) -> tuple[list[str], list[str]]:
    concepts = load_concepts(csv_path, limit=2 * per_split)
    return concepts[:per_split], concepts[per_split:]


def png_files(directory: str) -> list[str]:
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(item.name for item in path.iterdir() if item.suffix.lower() == ".png")


def concept_from_filename(filename: str, concepts: list[str]) -> str | None:
    """Recover a flat-layout concept label from a prompt-seed filename."""
    stem = Path(filename).stem.replace("_", " ")
    return next(
        (concept for concept in sorted(concepts, key=len, reverse=True) if concept in stem),
        None,
    )


def require_concept_directories(root: str, concepts: list[str], subdir: str) -> None:
    missing = [concept for concept in concepts if not (Path(root) / concept / subdir).is_dir()]
    if missing:
        preview = ", ".join(repr(item) for item in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise RuntimeError(
            f"Missing {len(missing)}/{len(concepts)} concept directories under {root}: "
            f"{preview}{suffix}"
        )
