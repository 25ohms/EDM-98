import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edm98")
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser("predict", help="Run inference on one audio file.")
    predict.add_argument("audio_path")

    predict_batch = subparsers.add_parser(
        "predict-batch", help="Run inference on a directory or manifest."
    )
    predict_batch.add_argument("input_path")

    validate = subparsers.add_parser(
        "validate-dataset", help="Validate an EDM-98 dataset JSONL file."
    )
    validate.add_argument("dataset_path")

    subparsers.add_parser("demo", help="Launch the Gradio demo.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    raise SystemExit(
        "CLI scaffold created. Implementation pending; see AGENTS.md for the plan."
    )
