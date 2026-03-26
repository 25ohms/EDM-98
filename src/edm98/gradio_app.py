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


def _build_waveform_html(audio_path: Path, prediction: list[dict[str, float | str]]) -> str:
    audio_data_url = _build_audio_data_url(audio_path)
    regions = []
    for segment in prediction:
        label = str(segment["label"])
        color = LABEL_COLORS.get(label, "#6C757D")
        regions.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "content": label.replace("_", " ").title(),
                "color": f"{color}66",
                "drag": False,
                "resize": False,
            }
        )

    html_id = f"waveform-{abs(hash((audio_path.name, tuple((r['start'], r['end'], r['content']) for r in regions))))}"
    regions_json = json.dumps(regions)

    return f"""
<div class="edm98-waveform-shell">
  <div class="edm98-toolbar">
    <button id="{html_id}-play" class="edm98-play">Play / Pause</button>
    <div class="edm98-time"><span id="{html_id}-current">0:00</span> / <span id="{html_id}-total">0:00</span></div>
  </div>
  <div id="{html_id}" class="edm98-waveform"></div>
</div>
<style>
  .edm98-waveform-shell {{
    width: 100%;
    border: 1px solid #d7dde5;
    border-radius: 20px;
    padding: 18px 18px 10px;
    background: linear-gradient(180deg, #fcfdff 0%, #eef4fb 100%);
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
  .edm98-waveform {{
    width: 100%;
    min-height: 220px;
  }}
  .edm98-time {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.95rem;
    color: #334155;
  }}
  .edm98-waveform ::-webkit-scrollbar {{
    display: none;
  }}
  .edm98-waveform region {{
    border-radius: 12px;
    overflow: hidden;
  }}
  .edm98-waveform region div {{
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 8px !important;
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
    height: 220,
    barWidth: 2,
    barGap: 1,
    fillParent: true,
    normalize: true,
    minPxPerSec: 120,
    autoScroll: true,
    autoCenter: true,
    hideScrollbar: true,
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
