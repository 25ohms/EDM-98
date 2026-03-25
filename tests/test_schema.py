from edm98.loaders import load_all_splits, load_dataset_records, load_records_by_split
from edm98.schema import validate_splits_against_dataset, validate_track_record


def test_dataset_loads() -> None:
    records = load_dataset_records()
    assert len(records) == 98


def test_splits_validate_against_dataset() -> None:
    records = load_dataset_records()
    splits = load_all_splits()
    summary = validate_splits_against_dataset(records, splits)
    assert summary.split_sizes == {"train": 78, "test": 10, "val": 10}


def test_split_loader_filters_records() -> None:
    train_records = load_records_by_split("train")
    test_records = load_records_by_split("test")
    val_records = load_records_by_split("val")
    assert len(train_records) == 78
    assert len(test_records) == 10
    assert len(val_records) == 10


def test_track_requires_terminal_end() -> None:
    try:
        validate_track_record(
            {"id": "x", "labels": [[0.0, "intro"], [1.0, "drop"]]},
            line_number=1,
        )
    except ValueError as exc:
        assert "terminate with 'end'" in str(exc)
    else:
        raise AssertionError("expected validation failure")
