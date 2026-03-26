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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    stripped = hex_color.lstrip("#")
    return tuple(int(stripped[idx : idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, value)):02x}" for value in rgb)


def _mix_hex(base: str, target: str, ratio: float) -> str:
    base_rgb = _hex_to_rgb(base)
    target_rgb = _hex_to_rgb(target)
    mixed = tuple(
        round(base_rgb[idx] * (1 - ratio) + target_rgb[idx] * ratio) for idx in range(3)
    )
    return _rgb_to_hex(mixed)


def _darken_hex(color: str, amount: float = 0.28) -> str:
    return _mix_hex(color, "#0f172a", amount)


def _lighten_hex(color: str, amount: float = 0.22) -> str:
    return _mix_hex(color, "#ffffff", amount)


def _format_clock(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes = whole_seconds // 60
    remainder = whole_seconds % 60
    return f"{minutes}:{remainder:02d}"


def _build_segment_table_html(prediction: list[dict[str, float | str]]) -> str:
    rows = []
    for segment in prediction:
        label = str(segment["label"]).replace("_", " ").title()
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start
        rows.append(
            f"""
            <tr>
              <td>{label}</td>
              <td>{start:.3f}s <span>({_format_clock(start)})</span></td>
              <td>{end:.3f}s <span>({_format_clock(end)})</span></td>
              <td>{duration:.3f}s <span>({_format_clock(duration)})</span></td>
            </tr>
            """
        )

    return f"""
    <div class="edm98-table-card">
      <div class="edm98-table-title">Predicted Segments</div>
      <div class="edm98-table-shell">
        <table class="edm98-table">
          <thead>
            <tr>
              <th>Label</th>
              <th>Start</th>
              <th>End</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def _build_audio_data_url(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(audio_path.name)
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_waveform_svg(audio_path: Path, prediction: list[dict[str, float | str]]) -> tuple[str, float, str]:
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

    run_lengths: list[int] = []
    run_positions: list[int] = []
    idx = 0
    while idx < len(prediction):
        label = str(prediction[idx]["label"])
        jdx = idx
        while jdx < len(prediction) and str(prediction[jdx]["label"]) == label:
            jdx += 1
        run_length = jdx - idx
        for position in range(run_length):
            run_lengths.append(run_length)
            run_positions.append(position)
        idx = jdx

    region_rects = []
    legend_items = []
    for segment, run_length, run_position in zip(prediction, run_lengths, run_positions):
        start = float(segment["start"])
        end = float(segment["end"])
        label = str(segment["label"])
        base_color = LABEL_COLORS.get(label, "#6C757D")
        if run_length > 1:
            ratio = run_position / max(run_length - 1, 1)
            fill_color = _mix_hex(
                _lighten_hex(base_color, 0.24),
                _darken_hex(base_color, 0.12),
                ratio,
            )
        else:
            fill_color = _lighten_hex(base_color, 0.18)
        border_color = _darken_hex(base_color, 0.32)
        start_x = 0 if duration == 0 else (start / duration) * width
        region_width = 0 if duration == 0 else max(((end - start) / duration) * width, 2)
        region_rects.append(
            f'<rect x="{start_x:.2f}" y="18" width="{region_width:.2f}" height="{height - 36}" '
            f'rx="14" ry="14" fill="{fill_color}" opacity="0.34" stroke="{border_color}" stroke-width="2"></rect>'
        )
        legend_items.append(
            f"<span class='edm98-legend-chip'><span class='edm98-legend-swatch' "
            f"style='background:{base_color}; border-color:{border_color}'></span>"
            f"{label.replace('_', ' ').title()}</span>"
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
            f'rx="{bar_width / 2:.2f}" ry="{bar_width / 2:.2f}" fill="#0f172a" opacity="0.9"></rect>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" ry="18" fill="#f8fbff"></rect>'
        f'{"".join(region_rects)}'
        f'{"".join(bars)}'
        "</svg>"
    )
    legend_html = "".join(dict.fromkeys(legend_items))
    return svg, duration, legend_html


def _player_head() -> str:
    return """
<script>
window.edm98Players = window.edm98Players || {};
window.edm98InitPlayer = function (id) {
  const root = document.getElementById(id);
  if (!root || window.edm98Players[id]) return;

  const audio = root.querySelector("audio");
  const button = root.querySelector("[data-role='toggle']");
  const stage = root.querySelector("[data-role='stage']");
  const playhead = root.querySelector("[data-role='playhead']");
  const current = root.querySelector("[data-role='current']");
  const total = root.querySelector("[data-role='total']");

  if (!audio || !button || !stage || !playhead || !current || !total) return;

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

  stage.addEventListener("click", (event) => {
    const rect = stage.getBoundingClientRect();
    const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
    const clamped = Math.min(Math.max(ratio, 0), 1);
    if (audio.duration) {
      audio.currentTime = clamped * audio.duration;
      sync();
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
    waveform_svg, duration, legend_html = _build_waveform_svg(audio_path, prediction)
    html_id = f"waveform-{abs(hash((audio_path.name, tuple((row['label'], row['start'], row['end']) for row in prediction))))}"

    return f"""
<div id="{html_id}" class="edm98-waveform-shell" data-player-id="{html_id}">
  <div class="edm98-toolbar">
    <button data-role="toggle" class="edm98-play">Play</button>
    <div class="edm98-time"><span data-role="current">0:00</span> / <span data-role="total">{_format_clock(duration)}</span></div>
  </div>
  <div class="edm98-waveform-stage" data-role="stage">
    <div class="edm98-waveform-svg">{waveform_svg}</div>
    <div class="edm98-playhead" data-role="playhead"></div>
  </div>
  <div class="edm98-legend">{legend_html}</div>
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
    margin-bottom: 14px;
  }}
  .edm98-play {{
    border: none;
    border-radius: 999px;
    background: #111827;
    color: white;
    padding: 10px 18px;
    font-weight: 700;
    font-family: Helvetica, Arial, sans-serif;
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
    cursor: pointer;
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
    font-family: Helvetica, Arial, sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #334155;
  }}
  .edm98-waveform-shell audio {{
    width: 100%;
    margin-top: 14px;
    accent-color: #111827;
  }}
  .edm98-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
  }}
  .edm98-legend-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid rgba(71, 85, 105, 0.34);
    color: #020617;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 0.92rem;
    font-weight: 800;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
  }}
  .edm98-legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    border: 2px solid transparent;
    display: inline-block;
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

    def run_inference(audio_file, progress=gr.Progress(track_tqdm=False)):
        if not audio_file:
            raise gr.Error("Upload an audio file before running inference.")

        audio_path = Path(audio_file)
        prediction = pipeline.predict_file(audio_path, progress_callback=progress)
        return (
            _build_waveform_html(audio_path, prediction),
            _build_segment_table_html(prediction),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )

    def reset_demo():
        return (
            "",
            "",
            gr.update(value="", visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
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
        :root {
          color-scheme: light dark;
          --edm-bg: #0f1115;
          --edm-fg: #f5f7fb;
          --edm-muted: #cbd5e1;
          --edm-card-bg: rgba(17, 24, 39, 0.92);
          --edm-card-border: rgba(148, 163, 184, 0.18);
          --edm-table-bg: rgba(15, 23, 42, 0.96);
          --edm-table-head: rgba(15, 23, 42, 0.7);
          --edm-table-row-a: rgba(30, 41, 59, 0.24);
          --edm-table-row-b: rgba(15, 23, 42, 0.16);
        }
        @media (prefers-color-scheme: light) {
          :root {
            --edm-bg: #f3f6fb;
            --edm-fg: #0f172a;
            --edm-muted: #475569;
            --edm-card-bg: rgba(255, 255, 255, 0.88);
            --edm-card-border: rgba(148, 163, 184, 0.24);
            --edm-table-bg: rgba(255, 255, 255, 0.92);
            --edm-table-head: rgba(241, 245, 249, 0.96);
            --edm-table-row-a: rgba(248, 250, 252, 0.98);
            --edm-table-row-b: rgba(241, 245, 249, 0.92);
          }
        }
        .gradio-container {
          max-width: min(1440px, 96vw) !important;
          font-family: Helvetica, Arial, sans-serif !important;
          margin: 0 auto !important;
        }
        .gradio-container, body {
          background: var(--edm-bg) !important;
          color: var(--edm-fg) !important;
        }
        h1, h2, h3 {
          font-family: Helvetica, Arial, sans-serif !important;
          font-weight: 800 !important;
          color: var(--edm-fg) !important;
        }
        p, .prose, .gr-markdown, .gr-markdown p {
          color: var(--edm-muted) !important;
        }
        .edm98-page {
          display: flex;
          flex-direction: column;
          gap: 18px;
          align-items: center;
        }
        .edm98-intro {
          width: 100%;
          text-align: center;
          margin: 0 auto;
        }
        .edm98-upload-card, .edm98-results-card {
          width: 100%;
          border-radius: 24px;
          border: 1px solid var(--edm-card-border);
          background: var(--edm-card-bg);
          padding: 18px;
          box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
          box-sizing: border-box;
        }
        .edm98-results-card {
          background: transparent;
          border: none;
          padding: 0;
          box-shadow: none;
        }
        .edm98-results {
          width: 100%;
        }
        .edm98-run-row, .edm98-reset-row {
          display: flex;
          justify-content: center;
        }
        .edm98-status {
          width: 100%;
          text-align: center;
          font-family: Helvetica, Arial, sans-serif;
          font-weight: 700;
          color: var(--edm-muted);
          min-height: 24px;
        }
        .edm98-reset-row { margin-top: 10px; }
        .edm98-table-card {
          width: 100%;
          border-radius: 22px;
          overflow: hidden;
          background: var(--edm-table-bg);
          border: 1px solid var(--edm-card-border);
          box-sizing: border-box;
        }
        .edm98-table-title {
          padding: 16px 20px 10px;
          font-size: 1rem;
          font-weight: 800;
          font-family: Helvetica, Arial, sans-serif;
          color: var(--edm-fg);
        }
        .edm98-table-shell {
          overflow: hidden;
          border-top: 1px solid rgba(148, 163, 184, 0.14);
        }
        .edm98-table {
          width: 100%;
          border-collapse: collapse;
          font-family: Helvetica, Arial, sans-serif;
        }
        .edm98-table thead th {
          text-align: left;
          padding: 14px 18px;
          font-size: 0.9rem;
          font-weight: 800;
          color: var(--edm-muted);
          background: var(--edm-table-head);
        }
        .edm98-table tbody td {
          padding: 14px 18px;
          color: var(--edm-fg);
          border-top: 1px solid rgba(148, 163, 184, 0.1);
          font-size: 0.96rem;
        }
        .edm98-table tbody td span {
          color: var(--edm-muted);
          font-weight: 600;
        }
        .edm98-table tbody tr:nth-child(odd) td {
          background: var(--edm-table-row-a);
        }
        .edm98-table tbody tr:nth-child(even) td {
          background: var(--edm-table-row-b);
        }
        """,
    ) as demo:
        with gr.Column(elem_classes=["edm98-page"]):
            with gr.Column(elem_classes=["edm98-intro"]):
                gr.Markdown("# EDM-98 Inference Demo")
                gr.Markdown(description)

            with gr.Column(elem_classes=["edm98-upload-card"], visible=True) as upload_panel:
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Audio File",
                )
                status_text = gr.Markdown("", visible=False, elem_classes=["edm98-status"])
                with gr.Row(elem_classes=["edm98-run-row"]):
                    run_button = gr.Button("Run Inference", variant="primary")

            with gr.Column(elem_classes=["edm98-results-card"], visible=False) as results_panel:
                waveform_output = gr.HTML(elem_classes=["edm98-results"])
                segment_table = gr.HTML()
                with gr.Row(elem_classes=["edm98-reset-row"]):
                    reset_button = gr.Button("Choose Another Audio")

        run_button.click(
            fn=run_inference,
            inputs=[audio_input],
            outputs=[waveform_output, segment_table, status_text, upload_panel, results_panel, reset_button],
            show_progress="full",
        )
        reset_button.click(
            fn=reset_demo,
            inputs=[],
            outputs=[waveform_output, segment_table, status_text, upload_panel, results_panel, reset_button],
            show_progress="hidden",
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
    demo = demo.queue(default_concurrency_limit=1)
    return demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )


if __name__ == "__main__":
    launch_demo()
