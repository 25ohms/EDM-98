# EDM-98

`EDM-98` is a lightweight open-source repository for the EDM-98 dataset with optional EDMFormer inference tooling.

## Intended Scope

This repository is intended to contain:

- the EDM-98 dataset metadata and labels
- a lightweight default Python package for dataset access and validation
- optional inference tooling for users who want to run EDMFormer locally
- an optional CLI for prediction and dataset validation
- an optional Gradio demo for interactive use
- clear contributor-facing documentation

## Provenance

EDM-98 was created from a curated 98-song selection, Rekordbox cue point labeling, and an extracted dataset artifact.

The dataset artifact originally existed as a JSON file and was later converted to JSONL to satisfy the label-file format expected by the SongFormer architecture. The canonical source of truth for EDM-98 labels is therefore `dataset.jsonl`.

## Licensing

This repository uses separate licenses by component:

- repository code and model-related materials: CC BY 4.0
- dataset metadata and split files under `data/`: MIT

Attribution should be preserved for upstream SongFormer-derived materials and for EDM-CUE-derived dataset metadata.

## Planned Package Surface

The repository is being designed around two usage modes.

### Dataset-only

This should be the default lightweight experience.

Users should be able to:

- inspect `dataset.jsonl`
- inspect `train.txt` and `test.txt`
- validate dataset artifacts
- contribute schema and metadata changes

### Optional inference

Users who want to experiment with EDMFormer inference should be able to install extra dependencies and provide a checkpoint path.

Recommended artifact locations:

- `data/dataset.jsonl`
- `data/splits/train.txt`
- `data/splits/test.txt`
- `data/checkpoints/model.pt` as the default checkpoint convention
- `configs/edmformer.yaml` as the public inference config

The current optional inference path also expects MusicFM support plus MusicFM weight files such as:

- `data/checkpoints/msd_stats.json`
- `data/checkpoints/pretrained_msd.pt`

The optional inference environment also requires the upstream MuQ and MusicFM codebases.

- MuQ is installed as a Python dependency.
- MusicFM is source-only upstream, so the supported setup is to clone it locally under `third_party/musicfm` using the provided install script.

Large binary assets in `data/checkpoints/` should be tracked with Git LFS.

Typical setup:

```bash
pip install -e ".[inference]"
git clone https://github.com/minzwon/musicfm.git third_party/musicfm
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
```

or

```bash
./scripts/install_inference_deps.sh
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
```

Recommended first-time setup:

```bash
./scripts/install_inference_deps.sh
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
```

To launch the Gradio app, also install the UI extra:

```bash
pip install -e ".[ui]"
```

The repository does not track `third_party/musicfm`; the install script provisions it locally for inference use.

### Local validation

Dataset-only validation:

```bash
pytest -q
PYTHONPATH=src python -m edm98.cli validate-dataset data/dataset.jsonl --splits-dir data/splits
```

Optional inference setup validation:

```bash
./scripts/install_inference_deps.sh
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
pytest -q
```

The optional inference test currently checks config-loading and dependency wiring. Real audio inference should be tested only after confirming that MuQ, MusicFM, `model.pt`, `pretrained_msd.pt`, and `msd_stats.json` are all available locally.

The checkpoint does not need to live in git. It can be hosted externally and documented in `data/checkpoints/README.md`.

The planned public interface is:

```bash
edm98 predict path/to/song.mp3
edm98 predict-batch path/to/input_dir
edm98 validate-dataset path/to/dataset.jsonl
edm98 demo
```

### Gradio demo

The Gradio app uses the same inference backend as `edm98 predict`.
It renders:

- a color-coded waveform timeline
- labeled section regions
- a moving playhead during playback
- the raw prediction JSON alongside the segment table

For local or Vertex AI Workbench use:

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
python -m edm98.cli demo --device cuda --low-memory --server-name 0.0.0.0 --server-port 7860
```

You can also launch it directly:

```bash
python -m edm98.gradio_app
```

In JupyterLab or Vertex AI Workbench, the app is typically reachable through the proxied port URL for `7860`. If needed, add `--share` to request a temporary public Gradio link.

## Next Work

- define the canonical public schema
- add the packaged inference core
- publish dataset and checkpoint documentation
