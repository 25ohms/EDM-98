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


def test_demo_cli_invokes_launcher(monkeypatch) -> None:
    called = {}

    def fake_launch_demo(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("edm98.gradio_app.launch_demo", fake_launch_demo)
    monkeypatch.setattr(
        "sys.argv",
        [
            "edm98",
            "demo",
            "--device",
            "cpu",
            "--low-memory",
            "--server-port",
            "9999",
        ],
    )
    code = main()
    assert code == 0
    assert called["device"] == "cpu"
    assert called["low_memory"] is True
    assert called["server_port"] == 9999
