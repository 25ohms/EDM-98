import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from edm98.constants import DEFAULT_DATA_DIR

from .labels import DATASET_LABEL_TO_DATASET_ID, build_label_mask
from .model import Model
from .postprocess import rule_post_processing


CONFIG_PATH = DEFAULT_DATA_DIR.parent / "configs" / "edmformer.yaml"
DEFAULT_CHECKPOINT_PATH = DEFAULT_DATA_DIR / "checkpoints" / "model.pt"
DEFAULT_MUSICFM_STAT_PATH = DEFAULT_DATA_DIR / "checkpoints" / "msd_stats.json"
DEFAULT_MUSICFM_MODEL_PATH = DEFAULT_DATA_DIR / "checkpoints" / "pretrained_msd.pt"

INPUT_SAMPLING_RATE = 24000
TIME_DUR = 420
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333


def _dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: _dict_to_namespace(value) for key, value in data.items()})
    if isinstance(data, list):
        return [_dict_to_namespace(item) for item in data]
    return data


def load_config(config_path: str | Path = CONFIG_PATH):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires PyYAML. Install optional inference dependencies first."
        ) from exc

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _dict_to_namespace(data)


def load_checkpoint(checkpoint_path: str | Path, device=None):
    checkpoint_path = str(checkpoint_path)
    if device is None:
        device = "cpu"

    if checkpoint_path.endswith(".pt"):
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return {"model_ema": load_file(checkpoint_path, device=device)}
    raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")


def _load_audio_backend():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires librosa. Install optional inference dependencies first."
        ) from exc
    return librosa


def _load_muq():
    try:
        from muq import MuQ
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires muq. Install optional inference dependencies first."
        ) from exc
    return MuQ


def _load_musicfm():
    try:
        module = importlib.import_module("musicfm.model.musicfm_25hz")
    except ImportError:
        third_party_root = DEFAULT_DATA_DIR.parent / "third_party"
        if third_party_root.exists() and str(third_party_root) not in sys.path:
            sys.path.insert(0, str(third_party_root))
        try:
            module = importlib.import_module("musicfm.model.musicfm_25hz")
        except ImportError as exc:
            raise RuntimeError(
                "Inference requires the MusicFM source tree. "
                "Clone https://github.com/minzwon/musicfm into third_party/musicfm "
                "or add its parent directory to PYTHONPATH, then retry."
            ) from exc
    return module.MusicFM25Hz


class InferencePipeline:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        config_path: str | Path = CONFIG_PATH,
        musicfm_stat_path: str | Path = DEFAULT_MUSICFM_STAT_PATH,
        musicfm_model_path: str | Path = DEFAULT_MUSICFM_MODEL_PATH,
        device: str | None = None,
        dataset_label: str = "EDMFormer",
        apply_rule_postprocessing: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = load_config(config_path)
        self.dataset_label = dataset_label
        self.apply_rule_postprocessing = apply_rule_postprocessing

        MuQ = _load_muq()
        MusicFM25Hz = _load_musicfm()

        self.muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq_model = self.muq_model.to(self.device).eval()

        self.musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=str(musicfm_stat_path),
            model_path=str(musicfm_model_path),
        )
        self.musicfm_model = self.musicfm_model.to(self.device).eval()

        self.model = Model(self.config)
        ckpt = load_checkpoint(checkpoint_path=checkpoint_path, device=self.device)
        if ckpt.get("model_ema") is not None:
            self.model.load_state_dict(ckpt["model_ema"], strict=True)
        elif ckpt.get("model") is not None:
            self.model.load_state_dict(ckpt["model"], strict=True)
        else:
            raise RuntimeError("Checkpoint missing 'model' or 'model_ema' state.")
        self.model.to(self.device).eval()

        mask = build_label_mask(
            num_classes=self.config.num_classes,
            dataset_label=self.dataset_label,
        )
        self.label_mask = (
            torch.tensor(mask, dtype=torch.bool, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.dataset_ids = torch.tensor(
            [DATASET_LABEL_TO_DATASET_ID[self.dataset_label]],
            dtype=torch.long,
            device=self.device,
        )

    def predict_file(self, audio_path: str | Path) -> list[dict[str, float | str]]:
        librosa = _load_audio_backend()
        wav, _sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
        audio = torch.tensor(wav).to(self.device)

        total_len = ((audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR * TIME_DUR) + TIME_DUR
        total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

        logits = {
            "function_logits": np.zeros([total_frames, self.config.num_classes]),
            "boundary_logits": np.zeros([total_frames]),
        }
        logits_num = {
            "function_logits": np.zeros([total_frames, self.config.num_classes]),
            "boundary_logits": np.zeros([total_frames]),
        }

        lens = 0
        i = 0

        with torch.no_grad():
            while True:
                start_idx = i * INPUT_SAMPLING_RATE
                end_idx = min((i + TIME_DUR) * INPUT_SAMPLING_RATE, audio.shape[-1])
                if start_idx >= audio.shape[-1]:
                    break
                if end_idx - start_idx <= 1024:
                    break

                audio_seg = audio[start_idx:end_idx]

                muq_output = self.muq_model(audio_seg.unsqueeze(0), output_hidden_states=True)
                muq_embd_420s = muq_output["hidden_states"][10]
                del muq_output

                _, musicfm_hidden_states = self.musicfm_model.get_predictions(
                    audio_seg.unsqueeze(0)
                )
                musicfm_embd_420s = musicfm_hidden_states[10]
                del musicfm_hidden_states

                wrapped_muq_embd_30s = []
                wrapped_musicfm_embd_30s = []

                for idx_30s in range(i, i + TIME_DUR, 30):
                    start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                    end_idx_30s = min(
                        (idx_30s + 30) * INPUT_SAMPLING_RATE,
                        audio.shape[-1],
                        (i + TIME_DUR) * INPUT_SAMPLING_RATE,
                    )
                    if start_idx_30s >= audio.shape[-1]:
                        break
                    if end_idx_30s - start_idx_30s <= 1024:
                        continue

                    wrapped_muq_embd_30s.append(
                        self.muq_model(
                            audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                            output_hidden_states=True,
                        )["hidden_states"][10]
                    )
                    wrapped_musicfm_embd_30s.append(
                        self.musicfm_model.get_predictions(
                            audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                        )[1][10]
                    )

                wrapped_muq_embd_30s = torch.concatenate(wrapped_muq_embd_30s, dim=1)
                wrapped_musicfm_embd_30s = torch.concatenate(
                    wrapped_musicfm_embd_30s, dim=1
                )

                all_embds = [
                    wrapped_musicfm_embd_30s,
                    wrapped_muq_embd_30s,
                    musicfm_embd_420s,
                    muq_embd_420s,
                ]

                embd_lens = [x.shape[1] for x in all_embds]
                min_embd_len = min(embd_lens)
                max_embd_len = max(embd_lens)
                if abs(max_embd_len - min_embd_len) > 4:
                    raise ValueError(
                        f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
                    )
                for idx in range(len(all_embds)):
                    all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                embd = torch.concatenate(all_embds, axis=-1)

                _msa_info, chunk_logits = self.model.infer(
                    input_embeddings=embd,
                    dataset_ids=self.dataset_ids,
                    label_id_masks=self.label_mask,
                    with_logits=True,
                )

                start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                end_frame = start_frame + min(
                    math.ceil(TIME_DUR * AFTER_DOWNSAMPLING_FRAME_RATES),
                    chunk_logits["boundary_logits"][0].shape[0],
                )

                logits["function_logits"][start_frame:end_frame, :] += (
                    chunk_logits["function_logits"][0].detach().cpu().numpy()
                )
                logits["boundary_logits"][start_frame:end_frame] = (
                    chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                )
                logits_num["function_logits"][start_frame:end_frame, :] += 1
                logits_num["boundary_logits"][start_frame:end_frame] += 1
                lens += end_frame - start_frame
                i += TIME_DUR

        logits["function_logits"] /= logits_num["function_logits"]
        logits["boundary_logits"] /= logits_num["boundary_logits"]

        logits["function_logits"] = torch.from_numpy(logits["function_logits"][:lens]).unsqueeze(0)
        logits["boundary_logits"] = torch.from_numpy(logits["boundary_logits"][:lens]).unsqueeze(0)

        from .postprocess import postprocess_functional_structure

        msa_infer_output = postprocess_functional_structure(logits, self.config)
        if self.apply_rule_postprocessing:
            msa_infer_output = rule_post_processing(msa_infer_output)

        output = []
        for idx in range(len(msa_infer_output) - 1):
            output.append(
                {
                    "label": msa_infer_output[idx][1],
                    "start": float(msa_infer_output[idx][0]),
                    "end": float(msa_infer_output[idx + 1][0]),
                }
            )
        return output


def predict_file(audio_path: str | Path, **kwargs):
    pipeline = InferencePipeline(**kwargs)
    return pipeline.predict_file(audio_path)


def write_prediction_json(prediction, output_path: str | Path):
    output_path = Path(output_path)
    output_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
