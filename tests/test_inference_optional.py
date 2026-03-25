import pytest


def test_inference_config_loads():
    pytest.importorskip("x_transformers")
    pytest.importorskip("omegaconf")
    pytest.importorskip("muq")
    pytest.importorskip("musicfm")

    from edm98.inference.pipeline import load_config

    cfg = load_config()
    assert cfg.input_dim_raw == 4096
    assert cfg.num_classes == 128
    assert cfg.slice_dur == 420
