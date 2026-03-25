from pathlib import Path

EDM98_LABELS = (
    "intro",
    "buildup",
    "drop",
    "breakdown",
    "outro",
    "silence",
)

TERMINATOR_LABEL = "end"
VALID_SEQUENCE_LABELS = EDM98_LABELS + (TERMINATOR_LABEL,)
DATASET_TYPE = "EDMFormer"

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "dataset.jsonl"
DEFAULT_SPLITS_DIR = DEFAULT_DATA_DIR / "splits"
