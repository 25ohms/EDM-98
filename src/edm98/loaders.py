import json
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_DATASET_PATH, DEFAULT_SPLITS_DIR
from .schema import TrackRecord, validate_dataset_records, validate_track_record


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def iter_dataset_records(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> Iterable[TrackRecord]:
    path = _coerce_path(dataset_path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield validate_track_record(payload, line_number=line_number)


def load_dataset_records(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[TrackRecord]:
    records = list(iter_dataset_records(dataset_path))
    validate_dataset_records(records)
    return records


def load_split_ids(
    split_name: str,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> list[str]:
    path = _coerce_path(splits_dir) / f"{split_name}.txt"
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_all_splits(
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> dict[str, list[str]]:
    return {
        split_name: load_split_ids(split_name, splits_dir=splits_dir)
        for split_name in ("train", "test", "val")
        if (_coerce_path(splits_dir) / f"{split_name}.txt").exists()
    }


def load_records_by_split(
    split_name: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> list[TrackRecord]:
    records = load_dataset_records(dataset_path)
    ids = set(load_split_ids(split_name, splits_dir=splits_dir))
    return [record for record in records if record["id"] in ids]
