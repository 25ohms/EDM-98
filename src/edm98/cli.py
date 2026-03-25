import argparse
import logging
from pathlib import Path

from .loaders import load_all_splits, load_dataset_records
from .schema import ValidationError, validate_splits_against_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edm98")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="Run inference on one audio file.")
    predict.add_argument("audio_path")
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--config", default=None)
    predict.add_argument("--musicfm-stat", default=None)
    predict.add_argument("--musicfm-model", default=None)
    predict.add_argument("--output", default=None)
    predict.add_argument(
        "--low-memory",
        action="store_true",
        help="Use a lower-memory blockwise inference path.",
    )
    predict.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection for inference.",
    )

    predict_batch = subparsers.add_parser(
        "predict-batch", help="Run inference on a directory or manifest."
    )
    predict_batch.add_argument("input_path")

    validate = subparsers.add_parser(
        "validate-dataset", help="Validate an EDM-98 dataset JSONL file."
    )
    validate.add_argument("dataset_path")
    validate.add_argument(
        "--splits-dir",
        default=None,
        help="Optional directory containing train.txt, test.txt, and val.txt.",
    )

    subparsers.add_parser("demo", help="Launch the Gradio demo.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "validate-dataset":
        try:
            records = load_dataset_records(args.dataset_path)
            splits = load_all_splits(args.splits_dir) if args.splits_dir else {}
            if splits:
                summary = validate_splits_against_dataset(records, splits)
                print(
                    f"Valid dataset: {summary.num_records} records, "
                    f"splits={summary.split_sizes}"
                )
            else:
                print(f"Valid dataset: {len(records)} records")
            return 0
        except (ValidationError, FileNotFoundError) as exc:
            print(f"Validation failed: {exc}")
            return 2

    if args.command == "predict":
        from .inference import predict_file

        kwargs = {}
        if args.checkpoint:
            kwargs["checkpoint_path"] = args.checkpoint
        if args.config:
            kwargs["config_path"] = args.config
        if args.musicfm_stat:
            kwargs["musicfm_stat_path"] = args.musicfm_stat
        if args.musicfm_model:
            kwargs["musicfm_model_path"] = args.musicfm_model
        kwargs["device"] = args.device
        kwargs["low_memory"] = args.low_memory
        prediction = predict_file(args.audio_path, **kwargs)
        if args.output:
            Path(args.output).write_text(
                __import__("json").dumps(prediction, indent=2), encoding="utf-8"
            )
        else:
            print(__import__("json").dumps(prediction, indent=2))
        return 0

    raise SystemExit("CLI command not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
