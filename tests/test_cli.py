from edm98.cli import main


def test_validate_dataset_cli(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "edm98",
            "validate-dataset",
            "data/dataset.jsonl",
            "--splits-dir",
            "data/splits",
        ],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Valid dataset" in out
