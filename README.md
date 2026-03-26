# EDM-98

`EDM-98` packages the EDM-98 dataset and an optional EDMFormer-based inference stack for local experimentation, app development, and downstream tooling.

## What Is Included

- the canonical EDM-98 label artifact packaged with the module
- canonical split files packaged with the module
- a lightweight Python package for dataset loading and validation
- an optional inference pipeline for EDMFormer
- a CLI for validation, prediction, cache warming, and demo launch
- a Gradio app with a waveform timeline and color-coded section predictions

## Dataset

EDM-98 was created from a curated 98-song set with Rekordbox cue-point labeling. The original dataset artifact was created as JSON and later converted to JSONL to match the label-file format expected by the SongFormer architecture.

The dataset and split files are loaded from packaged resources inside `edm98`, under `src/edm98/resources/`. That packaged copy is the canonical source used by the Python API and PyPI distribution.

The primary labels exposed by the EDMFormer setup are:

- `intro`
- `buildup`
- `drop`
- `breakdown`
- `outro`
- `silence`

Each packaged dataset record currently includes:

- `id`: the Deezer track identifier used as the canonical record ID
- `labels`: a strictly increasing list of `[time, label]` pairs terminated by `end`
- `file_path`: the original filename used during labeling when available

For local preprocessing and training, the canonical audio contract is that each downloaded song is stored as `<deezer_id>.<ext>`, for example `1060564312.mp3`. `file_path` is preserved as provenance metadata, not as the primary lookup key.

The package does not redistribute the audio itself. The Deezer IDs are included so users can map the metadata back to externally downloaded audio.

## Accessing The Dataset

Load the canonical packaged dataset:

```python
from edm98.loaders import load_dataset_records, load_all_splits, load_records_by_split

records = load_dataset_records()
splits = load_all_splits()
train_records = load_records_by_split("train")
```

Example record shape:

```python
{
    "id": "1060564312",
    "labels": [
        (0.054, "intro"),
        (35.942, "buildup"),
        (58.38, "silence"),
        (62.866, "drop"),
        ...
        (247.0, "end"),
    ],
    "file_path": "01 - Oak - Airwalk.mp3",
}
```

If you have downloaded the corresponding audio externally, you can join the metadata back to a local music directory by Deezer ID. For example:

```python
from pathlib import Path

from edm98.loaders import load_dataset_records

audio_dir = Path("/path/to/downloaded/audio")
records = load_dataset_records()
extensions = (".mp3", ".wav", ".flac", ".m4a")

for record in records:
    for ext in extensions:
        candidate = audio_dir / f"{record['id']}{ext}"
        if candidate.exists():
            print(record["id"], candidate, record["labels"][:3])
            break
```

This assumes you have already acquired the audio separately. `edm98` provides the labels, IDs, and split definitions; it does not fetch or ship the songs.

## Training Preparation

`edm98` is primarily a dataset package, but the packaged labels and split files can also be used as the canonical dataset-side inputs for EDMFormer-style training.

The repository includes a simple notebook at `notebooks/edm98_training_prep.ipynb` that shows:

- how to load the packaged metadata and split IDs
- how to map dataset records back to externally downloaded audio
- how to structure the four embedding directories EDMFormer expects
- how to construct the minimal train/eval dataset configuration

The audio itself is still external. A typical flow is:

1. Install `edm98` and load the packaged records.
2. Download the songs separately using the provided Deezer IDs and store them as `<deezer_id>.<ext>`.
3. Generate the MuQ and MusicFM embeddings required by EDMFormer.
4. Point your training configuration at the packaged JSONL labels and split files.

Minimal example:

```python
from pathlib import Path

from edm98.loaders import load_records_by_split

audio_dir = Path("/path/to/downloaded/audio")
train_records = load_records_by_split("train")
extensions = (".mp3", ".wav", ".flac", ".m4a")

resolved = []
for record in train_records:
    for ext in extensions:
        candidate = audio_dir / f"{record['id']}{ext}"
        if candidate.exists():
            resolved.append(
                {
                    "id": record["id"],
                    "audio_path": candidate,
                    "labels": record["labels"],
                }
            )
            break

resolved[:2]
```

That resolved list is the starting point for a preprocessing step that generates the EDMFormer-compatible MuQ and MusicFM embedding directories used during training.

## Installation

### Dataset-only

```bash
pip install edm98
```

### Inference

```bash
git clone https://github.com/25ohms/EDM-98.git
cd EDM-98
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export MUSICFMPATH="$PWD/third_party/musicfm"
```

`third_party/musicfm` is provisioned locally by the install script because upstream MusicFM is not published as an installable Python package. The same script also installs MuQ from its upstream source repository. Set `MUSICFMPATH` to that checkout when using the optional local inference workflow.

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
python -m edm98.cli validate-dataset
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
export MUSICFMPATH="$PWD/third_party/musicfm"
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
export MUSICFMPATH="$PWD/third_party/musicfm"
python -m edm98.cli demo --device cuda --server-name 0.0.0.0 --server-port 7860
```

### macOS

On Apple Silicon, use Metal via `--device mps`.

- Apple Silicon demo launch:

```bash
./scripts/install_inference_deps.sh
pip install -e ".[ui]"
export MUSICFMPATH="$PWD/third_party/musicfm"
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
python -m edm98.cli validate-dataset
```

Test suite:

```bash
pytest -q
```

## Licensing

This repository uses separate licenses by component:

- repository code and model-related materials: CC BY 4.0
- packaged dataset metadata and split files: MIT
