# EDM-98

`EDM-98` packages the EDM-98 dataset and an optional EDMFormer-based inference stack for local experimentation, app development, and downstream tooling.

## What Is Included

- the canonical EDM-98 label artifact at `data/dataset.jsonl`
- canonical split files in `data/splits/`
- a lightweight Python package for dataset loading and validation
- an optional inference pipeline for EDMFormer
- a CLI for validation, prediction, cache warming, and demo launch
- a Gradio app with a waveform timeline and color-coded section predictions

## Dataset

EDM-98 was created from a curated 98-song set with Rekordbox cue-point labeling. The original dataset artifact was created as JSON and later converted to JSONL to match the label-file format expected by the SongFormer architecture. The canonical source of truth for labels in this repository is `data/dataset.jsonl`.

The primary labels exposed by the EDMFormer setup are:

- `intro`
- `buildup`
- `drop`
- `breakdown`
- `outro`
- `silence`

## Installation

### Dataset-only

```bash
pip install -e .
```

### Inference

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
```

`third_party/musicfm` is provisioned locally by the install script because upstream MusicFM is not published as an installable Python package.

## Checkpoints And Cache

Expected local inference assets:

- `data/checkpoints/model.pt`
- `data/checkpoints/pretrained_msd.pt`
- `data/checkpoints/msd_stats.json`
- `configs/edmformer.yaml`

MuQ and MusicFM also depend on Hugging Face-backed upstream assets. Those are cached automatically under `.cache/huggingface/` on first use and reused on later runs.

Optional cache commands:

```bash
python -m edm98.cli warm-cache
python -m edm98.cli predict --offline path/to/song.mp3
python -m edm98.cli predict --no-cache path/to/song.mp3
```

## CLI

Validate the dataset:

```bash
python -m edm98.cli validate-dataset data/dataset.jsonl --splits-dir data/splits
```

Run inference on one file:

```bash
python -m edm98.cli predict --device cuda --low-memory path/to/song.mp3
```

Launch the Gradio demo:

```bash
python -m edm98.cli demo --device cuda --server-name 0.0.0.0 --server-port 7860
```

## Gradio Demo

The Gradio app uses the same inference backend as the CLI and preloads the inference pipeline when the app starts. That pipeline stays alive until the process exits, so the app does not rebuild the full EDMFormer, MuQ, and MusicFM stack for every request.

The demo is intentionally persistent. Start it once, keep the process running, and reuse the loaded pipeline until you close the app.

The demo currently provides:

- a file upload flow
- a full-width color-coded waveform timeline
- labeled section regions
- a moving playback cursor
- a tabular view of predicted sections with minute-second timestamps

To launch the demo:

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
python -m edm98.cli demo --device cuda --server-name 0.0.0.0 --server-port 7860
```

If you are running on a remote machine, expose or forward the chosen port and open the forwarded local URL in your browser.

### Demo Options

Useful demo flags:

- `--device auto`: pick the best available backend automatically
- `--device cuda`: run on an NVIDIA GPU
- `--device mps`: run on Apple Silicon via Metal
- `--device cpu`: force CPU inference
- `--server-name 0.0.0.0`: bind on all interfaces so you can forward or expose the port
- `--server-port 7860`: choose a different port if needed
- `--offline`: require Hugging Face-backed assets to already exist in the local cache
- `--no-cache`: use a temporary cache directory for this run
- `--hf-cache-dir <path>`: override the default Hugging Face cache location

`--low-memory` is useful for one-off CLI prediction runs, but it is not the intended mode for the Gradio demo. The demo is designed to keep its models resident until shutdown.

## Platform Notes

The CLI currently supports `--device auto`, `--device cpu`, `--device cuda`, and `--device mps`.

### Linux

Linux is the most straightforward setup for GPU-backed demo usage.

- NVIDIA GPU: use `--device cuda`
- CPU-only: use `--device cpu`
- Typical demo launch:

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
python -m edm98.cli demo --device cuda --server-name 0.0.0.0 --server-port 7860
```

### macOS

On Apple Silicon, use Metal via `--device mps`.

- Apple Silicon demo launch:

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export PYTHONPATH="$PWD/src:$PWD/third_party:$PYTHONPATH"
python -m edm98.cli demo --device mps --server-name 127.0.0.1 --server-port 7860
```

- If MPS is unavailable or unstable in your local environment, fall back to `--device cpu`

### Windows

The supported install helper in this repository is `scripts/install_inference_deps.sh`, which is a Bash script. Because of that, the smoothest Windows path is currently a Bash-compatible environment such as WSL2 or Git Bash, with WSL2 being the more predictable choice for ML dependencies.

- Windows + WSL2 + NVIDIA GPU: use `--device cuda`
- Windows + WSL2 CPU-only: use `--device cpu`
- If you want a browser on Windows to access a demo running inside WSL2, open the forwarded localhost URL from Windows after launch

If you are running a fully native Windows Python environment instead of WSL2, the same CLI flags apply, but you will need to reproduce the install-script steps manually.

## Python API

For one-off inference:

```python
from edm98.inference import predict_file

prediction = predict_file("song.mp3", device="cuda", low_memory=True)
```

For app integration or repeated use, create the pipeline once and reuse it:

```python
from edm98.inference import create_pipeline

pipeline = create_pipeline(
    device="cuda",
    persistent_models=True,
)

prediction = pipeline.predict_file("song.mp3")
```

This is the same pattern used by the Gradio app.

## Developer Notes

- `predict` is suitable for single-use command-line workflows.
- `InferencePipeline` is the stable object to reuse inside other applications.
- `create_pipeline(...)` is provided as a small convenience wrapper for app startup code.
- the current repo-local cache behavior is the default and should remain transparent to most users

## Validation

Dataset validation:

```bash
python -m edm98.cli validate-dataset data/dataset.jsonl --splits-dir data/splits
```

Test suite:

```bash
pytest -q
```

## Licensing

This repository uses separate licenses by component:

- repository code and model-related materials: CC BY 4.0
- dataset metadata and split files under `data/`: MIT
