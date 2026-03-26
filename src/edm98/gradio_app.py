import base64
import mimetypes
from pathlib import Path

import numpy as np


LABEL_COLORS = {
    "intro": "#4F7CAC",
    "buildup": "#F0A202",
    "drop": "#D7263D",
    "breakdown": "#2E294E",
    "outro": "#1B998B",
    "silence": "#A0A4B8",
}


def _format_clock(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remainder = whole_seconds % 60
    return f"{minutes}:{remainder:02d}"


def _format_segments(prediction: list[dict[str, float | str]]) -> list[list[str | float]]:
    rows = []
    for segment in prediction:
        start = float(segment["start"])
        end = float(segment["end"])
        rows.append(
            [
                str(segment["label"]).replace("_", " ").title(),
                f"{start:.3f}s ({_format_clock(start)})",
                f"{end:.3f}s ({_format_clock(end)})",
                f"{(end - start):.3f}s ({_format_clock(end - start)})",
            ]
        )
    return rows


def _build_audio_data_url(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(audio_path.name)
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_waveform_svg(audio_path: Path, prediction: list[dict[str, float | str]]) -> tuple[str, float]:
    import librosa

    samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    duration = 0.0 if sample_rate <= 0 else float(len(samples) / sample_rate)
    width = 1200
    height = 260
    center_y = height / 2
    usable_height = height * 0.72

    target_bars = 360
    if len(samples) == 0:
        peaks = np.zeros(target_bars, dtype=np.float32)
    else:
        trimmed = np.abs(samples[: (len(samples) // target_bars) * target_bars])
        if trimmed.size == 0:
            peaks = np.zeros(target_bars, dtype=np.float32)
        else:
            peaks = trimmed.reshape(target_bars, -1).max(axis=1)
            max_peak = float(np.max(peaks))
            if max_peak > 0:
                peaks = peaks / max_peak

    region_rects = []
    for segment in prediction:
        start = float(segment["start"])
        end = float(segment["end"])
        label = str(segment["label"])
        color = LABEL_COLORS.get(label, "#6C757D")
        start_x = 0 if duration == 0 else (start / duration) * width
        region_width = 0 if duration == 0 else max(((end - start) / duration) * width, 2)
        region_rects.append(
            f'<rect x="{start_x:.2f}" y="18" width="{region_width:.2f}" height="{height - 36}" '
            f'rx="14" ry="14" fill="{color}" opacity="0.24"></rect>'
        )

    bars = []
    step = width / max(len(peaks), 1)
    bar_width = max(step * 0.62, 1.5)
    for idx, peak in enumerate(peaks):
        bar_height = max(float(peak) * usable_height, 4)
        x = idx * step + (step - bar_width) / 2
        y = center_y - bar_height / 2
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'rx="{bar_width / 2:.2f}" ry="{bar_width / 2:.2f}" fill="#0f172a" opacity="0.88"></rect>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" ry="18" fill="#f8fbff"></rect>'
        f'{"".join(region_rects)}'
        f'{"".join(bars)}'
        "</svg>"
    )
    return svg, duration


def _player_head() -> str:
    return """
<script>
window.edm98Players = window.edm98Players || {};
window.edm98InitPlayer = function (id) {
  const root = document.getElementById(id);
  if (!root || window.edm98Players[id]) return;

  const audio = root.querySelector("audio");
  const button = root.querySelector("[data-role='toggle']");
  const playhead = root.querySelector("[data-role='playhead']");
  const current = root.querySelector("[data-role='current']");
  const total = root.querySelector("[data-role='total']");

  if (!audio || !button || !playhead || !current || !total) return;

  const formatTime = (seconds) => {
    const safe = Math.max(0, Number(seconds || 0));
    const mins = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60).toString().padStart(2, "0");
    return `${mins}:${secs}`;
  };

  const sync = () => {
    const duration = Number(audio.duration || 0);
    const currentTime = Number(audio.currentTime || 0);
    total.textContent = formatTime(duration);
    current.textContent = formatTime(currentTime);
    const ratio = duration > 0 ? Math.min(currentTime / duration, 1) : 0;
    playhead.style.left = `${ratio * 100}%`;
  };

  button.addEventListener("click", () => {
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  });

  audio.addEventListener("loadedmetadata", sync);
  audio.addEventListener("timeupdate", sync);
  audio.addEventListener("pause", () => { button.textContent = "Play"; sync(); });
  audio.addEventListener("play", () => { button.textContent = "Pause"; sync(); });
  audio.addEventListener("ended", () => { button.textContent = "Play"; sync(); });

  sync();
  window.edm98Players[id] = true;
};

window.edm98ScanPlayers = function () {
  document.querySelectorAll(".edm98-waveform-shell[data-player-id]").forEach((node) => {
    const id = node.getAttribute("data-player-id");
    if (id) window.edm98InitPlayer(id);
  });
};

if (!window.edm98PlayerScannerStarted) {
  window.edm98PlayerScannerStarted = true;
  window.addEventListener("load", window.edm98ScanPlayers);
  setInterval(window.edm98ScanPlayers, 500);
}
</script>
"""


def _build_waveform_html(audio_path: Path, prediction: list[dict[str, float | str]]) -> str:
    audio_data_url = _build_audio_data_url(audio_path)
    waveform_svg, duration = _build_waveform_svg(audio_path, prediction)
    html_id = f"waveform-{abs(hash((audio_path.name, tuple((row['label'], row['start'], row['end']) for row in prediction))))}"

    return f"""
<div id="{html_id}" class="edm98-waveform-shell" data-player-id="{html_id}">
  <div class="edm98-toolbar">
    <button data-role="toggle" class="edm98-play">Play</button>
    <div class="edm98-time"><span data-role="current">0:00</span> / <span data-role="total">{_format_clock(duration)}</span></div>
  </div>
  <div class="edm98-waveform-stage">
    <div class="edm98-waveform-svg">{waveform_svg}</div>
    <div class="edm98-playhead" data-role="playhead"></div>
  </div>
  <audio preload="metadata" src="{audio_data_url}"></audio>
</div>
<style>
  .edm98-waveform-shell {{
    width: 100%;
    border: 1px solid #d7dde5;
    border-radius: 22px;
    padding: 18px;
    background: linear-gradient(180deg, #fbfdff 0%, #edf3fb 100%);
    box-sizing: border-box;
    overflow: hidden;
  }}
  .edm98-toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
  }}
  .edm98-play {{
    border: none;
    border-radius: 999px;
    background: #111827;
    color: white;
    padding: 10px 16px;
    font-weight: 600;
    cursor: pointer;
  }}
  .edm98-waveform-stage {{
    position: relative;
    width: 100%;
    min-height: 260px;
    overflow: hidden;
    border-radius: 18px;
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.18);
    background: linear-gradient(180deg, rgba(255,255,255,0.86) 0%, rgba(239,245,252,0.96) 100%);
  }}
  .edm98-waveform-svg {{
    width: 100%;
    height: 260px;
  }}
  .edm98-waveform-svg svg {{
    display: block;
    width: 100%;
    height: 260px;
  }}
  .edm98-playhead {{
    position: absolute;
    top: 12px;
    bottom: 12px;
    left: 0%;
    width: 2px;
    background: linear-gradient(180deg, #ef4444 0%, #f97316 100%);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.32);
    pointer-events: none;
    transform: translateX(-1px);
  }}
  .edm98-time {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.95rem;
    color: #334155;
  }}
  .edm98-waveform-shell audio {{
    width: 100%;
    margin-top: 14px;
    accent-color: #111827;
  }}
</style>
"""


def build_demo(
    *,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    musicfm_stat_path: str | None = None,
    musicfm_model_path: str | None = None,
    device: str = "auto",
    low_memory: bool = False,
    hf_cache_dir: str | None = None,
    offline: bool = False,
    no_cache: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the demo. Install optional UI dependencies first."
        ) from exc

    from .inference import create_pipeline

    pipeline_kwargs = {
        "device": device,
        "low_memory": low_memory,
        "persistent_models": True,
        "offline": offline,
        "no_cache": no_cache,
    }
    if checkpoint_path is not None:
        pipeline_kwargs["checkpoint_path"] = checkpoint_path
    if config_path is not None:
        pipeline_kwargs["config_path"] = config_path
    if musicfm_stat_path is not None:
        pipeline_kwargs["musicfm_stat_path"] = musicfm_stat_path
    if musicfm_model_path is not None:
        pipeline_kwargs["musicfm_model_path"] = musicfm_model_path
    if hf_cache_dir is not None:
        pipeline_kwargs["hf_cache_dir"] = hf_cache_dir

    pipeline = create_pipeline(**pipeline_kwargs)

    def run_inference(audio_file):
        if not audio_file:
            raise gr.Error("Upload an audio file before running inference.")

        audio_path = Path(audio_file)
        prediction = pipeline.predict_file(audio_path)
        return (
            _format_segments(prediction),
            _build_waveform_html(audio_path, prediction),
        )

    description = (
        "Upload an audio file, run EDMFormer inference, and inspect the predicted "
        "EDM-98 segment timeline. The waveform view is color-coded by section and "
        "includes a moving playhead during playback. The inference pipeline is "
        "preloaded when the app starts and remains live until the process exits."
    )

    with gr.Blocks(
        title="EDM-98 Demo",
        head=_player_head(),
        css="""
        .gradio-container {max-width: min(1600px, 98vw) !important;}
        .edm98-results {width: 100%;}
        .edm98-table .wrap.svelte-1ipelgc {font-size: 0.95rem;}
        """,
    ) as demo:
        gr.Markdown("# EDM-98 Inference Demo")
        gr.Markdown(description)
        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Audio File",
        )
        run_button = gr.Button("Run Inference", variant="primary")
        waveform_output = gr.HTML(elem_classes=["edm98-results"])
        segment_table = gr.Dataframe(
            headers=["Label", "Start", "End", "Duration"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            label="Predicted Segments",
            elem_classes=["edm98-table"],
        )

        run_button.click(
            fn=run_inference,
            inputs=[audio_input],
            outputs=[segment_table, waveform_output],
        )

    return demo


def launch_demo(
    *,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    musicfm_stat_path: str | None = None,
    musicfm_model_path: str | None = None,
    device: str = "auto",
    low_memory: bool = False,
    hf_cache_dir: str | None = None,
    offline: bool = False,
    no_cache: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
):
    demo = build_demo(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        musicfm_stat_path=musicfm_stat_path,
        musicfm_model_path=musicfm_model_path,
        device=device,
        low_memory=low_memory,
        hf_cache_dir=hf_cache_dir,
        offline=offline,
        no_cache=no_cache,
    )
    return demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_demo()
