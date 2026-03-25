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

It is intentionally not intended to contain the full cloud training and orchestration setup used during model development.

## Status

This repository has been initialized as the productionization target for the existing work spread across:

- `EDM-CUE_testing`
- `EDMFormer`
- `EDMFormer-project`

The implementation plan lives in [AGENTS.md](/Users/sahal/Desktop/ohms/code/WATAI/EDM-98/AGENTS.md).

## Provenance

EDM-98 was created from a curated 98-song selection, Rekordbox cue point labeling, and an extracted dataset artifact.

The dataset artifact originally existed as a JSON file and was later converted to JSONL to satisfy the label-file format expected by the SongFormer architecture. The canonical source of truth for EDM-98 labels is therefore `dataset.jsonl`.

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
- `data/checkpoints/EDMFormer.safetensors` as the default local convention

The checkpoint does not need to live in git. It can be hosted externally and documented in `data/checkpoints/README.md`.

The planned public interface is:

```bash
edm98 predict path/to/song.mp3
edm98 predict-batch path/to/input_dir
edm98 validate-dataset path/to/dataset.jsonl
edm98 demo
```

## Next Work

- add `dataset.jsonl`, `train.txt`, and `test.txt`
- define the canonical public schema
- add the packaged inference core
- add the CLI and Gradio app
- publish dataset and checkpoint documentation
