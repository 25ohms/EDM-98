import json
from importlib import resources
from pathlib import Path
from typing import Iterable

from .constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SPLITS_DIR,
    PACKAGE_RESOURCES_PACKAGE,
    PACKAGE_SPLITS_PACKAGE,
)
from .schema import TrackRecord, validate_dataset_records, validate_track_record


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def iter_dataset_records(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> Iterable[TrackRecord]:
    if dataset_path is None or dataset_path == DEFAULT_DATASET_PATH:
        resource = resources.files(PACKAGE_RESOURCES_PACKAGE).joinpath(DEFAULT_DATASET_PATH)
        handle_cm = resource.open("r", encoding="utf-8")
    else:
        handle_cm = _coerce_path(dataset_path).open(encoding="utf-8")

    with handle_cm as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield validate_track_record(payload, line_number=line_number)


def load_dataset_records(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> list[TrackRecord]:
    records = list(iter_dataset_records(dataset_path))
    validate_dataset_records(records)
    return records


def load_split_ids(
    split_name: str,
    splits_dir: str | Path | None = DEFAULT_SPLITS_DIR,
) -> list[str]:
    if splits_dir is None:
        resource = resources.files(PACKAGE_SPLITS_PACKAGE).joinpath(f"{split_name}.txt")
        handle_cm = resource.open("r", encoding="utf-8")
    else:
        handle_cm = (_coerce_path(splits_dir) / f"{split_name}.txt").open(encoding="utf-8")

    with handle_cm as handle:
        return [line.strip() for line in handle if line.strip()]


def load_all_splits(
    splits_dir: str | Path | None = DEFAULT_SPLITS_DIR,
) -> dict[str, list[str]]:
    if splits_dir is None:
        available = {"train", "test", "val"}
    else:
        base = _coerce_path(splits_dir)
        available = {
            split_name
            for split_name in ("train", "test", "val")
            if (base / f"{split_name}.txt").exists()
        }
    return {
        split_name: load_split_ids(split_name, splits_dir=splits_dir)
        for split_name in ("train", "test", "val")
        if split_name in available
    }


def load_records_by_split(
    split_name: str,
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    splits_dir: str | Path | None = DEFAULT_SPLITS_DIR,
) -> list[TrackRecord]:
    records = load_dataset_records(dataset_path)
    ids = set(load_split_ids(split_name, splits_dir=splits_dir))
    return [record for record in records if record["id"] in ids]
