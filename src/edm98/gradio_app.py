import base64
import json
import mimetypes
from pathlib import Path


LABEL_COLORS = {
    "intro": "#4F7CAC",
    "buildup": "#F0A202",
    "drop": "#D7263D",
    "breakdown": "#2E294E",
    "outro": "#1B998B",
    "silence": "#A0A4B8",
}


def _format_segments(prediction: list[dict[str, float | str]]) -> list[list[str | float]]:
    rows = []
    for segment in prediction:
        start = float(segment["start"])
        end = float(segment["end"])
        rows.append(
            [
                str(segment["label"]),
                round(start, 3),
                round(end, 3),
                round(end - start, 3),
            ]
        )
    return rows


def _build_audio_data_url(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(audio_path.name)
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_waveform_html(audio_path: Path, prediction: list[dict[str, float | str]]) -> str:
    audio_data_url = _build_audio_data_url(audio_path)
    regions = []
    legend_items = []
    for segment in prediction:
        label = str(segment["label"])
        color = LABEL_COLORS.get(label, "#6C757D")
        regions.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "content": label,
                "color": f"{color}66",
            }
        )
        legend_items.append(
            f"<span class='legend-chip'><span class='legend-swatch' style='background:{color}'></span>{label}</span>"
        )

    html_id = f"waveform-{abs(hash((audio_path.name, tuple((r['start'], r['end'], r['content']) for r in regions))))}"
    regions_json = json.dumps(regions)
    legend_html = "".join(dict.fromkeys(legend_items))

    return f"""
<div class="edm98-waveform-shell">
  <div class="edm98-toolbar">
    <button id="{html_id}-play" class="edm98-play">Play / Pause</button>
    <div class="edm98-time"><span id="{html_id}-current">0:00</span> / <span id="{html_id}-total">0:00</span></div>
  </div>
  <div id="{html_id}" class="edm98-waveform"></div>
  <div class="edm98-legend">{legend_html}</div>
</div>
<style>
  .edm98-waveform-shell {{
    border: 1px solid #d7dde5;
    border-radius: 16px;
    padding: 16px;
    background: linear-gradient(180deg, #fcfdff 0%, #f4f7fb 100%);
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
  .edm98-waveform {{
    min-height: 180px;
  }}
  .edm98-time {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.95rem;
    color: #334155;
  }}
  .edm98-legend {{
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .legend-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #d7dde5;
    font-size: 0.9rem;
  }}
  .legend-swatch {{
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
  }}
</style>
<script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
<script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
<script>
(() => {{
  const container = document.getElementById("{html_id}");
  if (!container) return;

  const playButton = document.getElementById("{html_id}-play");
  const currentTime = document.getElementById("{html_id}-current");
  const totalTime = document.getElementById("{html_id}-total");

  const formatTime = (seconds) => {{
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${{mins}}:${{secs}}`;
  }};

  const regions = {regions_json};
  const regionsPlugin = WaveSurfer.Regions.create();
  const ws = WaveSurfer.create({{
    container,
    waveColor: "#94a3b8",
    progressColor: "#0f172a",
    cursorColor: "#ef4444",
    cursorWidth: 2,
    height: 180,
    barWidth: 2,
    barGap: 1,
    url: "{audio_data_url}",
    plugins: [regionsPlugin],
  }});

  ws.on("decode", (duration) => {{
    totalTime.textContent = formatTime(duration);
    regions.forEach((region) => regionsPlugin.addRegion(region));
  }});

  ws.on("timeupdate", (seconds) => {{
    currentTime.textContent = formatTime(seconds);
  }});

  playButton.addEventListener("click", () => ws.playPause());
  ws.on("play", () => {{ playButton.textContent = "Pause"; }});
  ws.on("pause", () => {{ playButton.textContent = "Play"; }});
  ws.on("finish", () => {{ playButton.textContent = "Play"; }});
}})();
</script>
"""


def build_demo(
    *,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    musicfm_stat_path: str | None = None,
    musicfm_model_path: str | None = None,
    device: str = "auto",
    low_memory: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the demo. Install optional UI dependencies first."
        ) from exc

    def run_inference(audio_file):
        from .inference import predict_file

        if not audio_file:
            raise gr.Error("Upload an audio file before running inference.")

        audio_path = Path(audio_file)
        kwargs = {
            "device": device,
            "low_memory": low_memory,
        }
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        if config_path is not None:
            kwargs["config_path"] = config_path
        if musicfm_stat_path is not None:
            kwargs["musicfm_stat_path"] = musicfm_stat_path
        if musicfm_model_path is not None:
            kwargs["musicfm_model_path"] = musicfm_model_path

        prediction = predict_file(audio_path, **kwargs)
        return (
            _format_segments(prediction),
            json.dumps(prediction, indent=2),
            _build_waveform_html(audio_path, prediction),
        )

    description = (
        "Upload an audio file, run EDMFormer inference, and inspect the predicted "
        "EDM-98 segment timeline. The waveform view is color-coded by section and "
        "includes a moving playhead during playback."
    )

    with gr.Blocks(title="EDM-98 Demo") as demo:
        gr.Markdown("# EDM-98 Inference Demo")
        gr.Markdown(description)
        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Audio File",
        )
        run_button = gr.Button("Run Inference", variant="primary")
        waveform_output = gr.HTML()
        segment_table = gr.Dataframe(
            headers=["Label", "Start (s)", "End (s)", "Duration (s)"],
            datatype=["str", "number", "number", "number"],
            interactive=False,
            label="Predicted Segments",
        )
        json_output = gr.Code(
            label="Prediction JSON",
            language="json",
            interactive=False,
        )

        run_button.click(
            fn=run_inference,
            inputs=[audio_input],
            outputs=[segment_table, json_output, waveform_output],
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
    )
    return demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_demo()
