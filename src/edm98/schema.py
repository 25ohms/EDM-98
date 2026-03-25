from dataclasses import dataclass
from pathlib import Path

from .constants import EDM98_LABELS, TERMINATOR_LABEL


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    num_records: int
    unique_ids: int
    split_sizes: dict[str, int]


LabelEntry = tuple[float, str]
TrackRecord = dict[str, str | list[LabelEntry]]


def _fail(message: str, *, line_number: int | None = None) -> None:
    if line_number is None:
        raise ValidationError(message)
    raise ValidationError(f"line {line_number}: {message}")


def validate_track_record(
    payload: object,
    *,
    line_number: int | None = None,
) -> TrackRecord:
    if not isinstance(payload, dict):
        _fail("record must be a JSON object", line_number=line_number)

    if "id" not in payload:
        _fail("missing required field 'id'", line_number=line_number)
    if "labels" not in payload:
        _fail("missing required field 'labels'", line_number=line_number)

    record_id = str(payload["id"]).strip()
    if not record_id:
        _fail("'id' must be a non-empty string", line_number=line_number)

    file_path = payload.get("file_path")
    if file_path is not None and (not isinstance(file_path, str) or not file_path.strip()):
        _fail("'file_path' must be a non-empty string when provided", line_number=line_number)

    labels = payload["labels"]
    if not isinstance(labels, list) or not labels:
        _fail("'labels' must be a non-empty list", line_number=line_number)

    normalized_labels: list[LabelEntry] = []
    last_time = None
    for idx, entry in enumerate(labels):
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            _fail(
                f"'labels[{idx}]' must be a [time, label] pair",
                line_number=line_number,
            )
        raw_time, raw_label = entry
        if not isinstance(raw_time, int | float):
            _fail(f"'labels[{idx}][0]' must be numeric", line_number=line_number)
        if not isinstance(raw_label, str) or not raw_label.strip():
            _fail(f"'labels[{idx}][1]' must be a non-empty string", line_number=line_number)

        time_value = float(raw_time)
        label_value = raw_label.strip()

        if time_value < 0:
            _fail(f"'labels[{idx}][0]' must be >= 0", line_number=line_number)
        if last_time is not None and time_value <= last_time:
            _fail("'labels' times must be strictly increasing", line_number=line_number)
        if label_value not in EDM98_LABELS and label_value != TERMINATOR_LABEL:
            _fail(f"unknown label '{label_value}'", line_number=line_number)
        if label_value == TERMINATOR_LABEL and idx != len(labels) - 1:
            _fail("'end' must only appear as the final label", line_number=line_number)

        normalized_labels.append((time_value, label_value))
        last_time = time_value

    if normalized_labels[-1][1] != TERMINATOR_LABEL:
        _fail("'labels' must terminate with 'end'", line_number=line_number)

    normalized: TrackRecord = {
        "id": record_id,
        "labels": normalized_labels,
    }
    if file_path is not None:
        normalized["file_path"] = file_path.strip()
    return normalized


def validate_dataset_records(records: list[TrackRecord]) -> None:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
        raise ValidationError(f"duplicate record ids found: {', '.join(duplicates[:5])}")


def validate_split_ids(
    split_name: str,
    split_ids: list[str],
    dataset_ids: set[str],
) -> None:
    if len(split_ids) != len(set(split_ids)):
        raise ValidationError(f"split '{split_name}' contains duplicate ids")
    missing = sorted(set(split_ids) - dataset_ids)
    if missing:
        raise ValidationError(
            f"split '{split_name}' references ids missing from dataset: {', '.join(missing[:5])}"
        )


def validate_splits_against_dataset(
    records: list[TrackRecord],
    splits: dict[str, list[str]],
) -> DatasetSummary:
    dataset_ids = {record["id"] for record in records}
    split_sizes: dict[str, int] = {}

    for split_name, split_ids in splits.items():
        validate_split_ids(split_name, split_ids, dataset_ids)
        split_sizes[split_name] = len(split_ids)

    seen_ids: dict[str, str] = {}
    for split_name, split_ids in splits.items():
        for record_id in split_ids:
            if record_id in seen_ids:
                raise ValidationError(
                    f"id '{record_id}' appears in both '{seen_ids[record_id]}' and '{split_name}'"
                )
            seen_ids[record_id] = split_name

    if dataset_ids != set(seen_ids):
        missing_from_splits = sorted(dataset_ids - set(seen_ids))
        extra_in_splits = sorted(set(seen_ids) - dataset_ids)
        if missing_from_splits:
            raise ValidationError(
                f"dataset ids missing from splits: {', '.join(missing_from_splits[:5])}"
            )
        if extra_in_splits:
            raise ValidationError(
                f"split ids missing from dataset: {', '.join(extra_in_splits[:5])}"
            )

    return DatasetSummary(
        num_records=len(records),
        unique_ids=len(dataset_ids),
        split_sizes=split_sizes,
    )
