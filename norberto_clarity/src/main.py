"""Train the selected NorBERTo model or fill an unlabeled Excel workbook."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from threadpoolctl import threadpool_limits

from clarity.finetune import NORBERTO_ID, read_finetune_config, run

DEFAULT_CONFIG = Path("configs/norberto-lora-development-ctx512.json")
DEFAULT_SPLITS = Path("results/research/dataset-hypotheses-20260923/splits.csv")
DEFAULT_EXPERIMENT = Path("results/norberto/delivery/norberto-lora-development-ctx512-20260928")
DEFAULT_ARTIFACT = Path("artifacts/norberto-lora-final-ctx512-20260928")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a fresh NorBERTo adapter on development rows only.")
    train.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    train.add_argument("--output-dir", type=Path, required=True,
                       help="New directory for training reports and metadata.")
    train.add_argument("--artifact-dir", type=Path, required=True,
                       help="New directory for checkpoints and the final adapter.")
    train.add_argument("--holdout-splits", type=Path, default=DEFAULT_SPLITS,
                       help="Frozen holdout split; its rows are excluded from training.")
    train.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT,
                       help="Completed three-fold experiment used to select the epoch count.")
    train.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                       help="Train on CPU or GPU; auto prefers GPU. GPU detects CUDA or ROCm.")

    test = commands.add_parser("test", help="Fill clarity in an unlabeled workbook; do not train.")
    test.add_argument("--test", type=Path, required=True,
                      help="Input workbook such as data/test1.xlsx.")
    test.add_argument("--output", type=Path,
                      help="New output workbook; defaults to deliveries/<input filename>.")
    test.add_argument("--output-dir", type=Path,
                      help="New directory for prediction reports and probabilities.")
    test.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT,
                      help="Directory containing the verified final_model adapter.")
    test.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                      help="Infer on CPU or GPU; auto prefers GPU. GPU detects CUDA or ROCm.")
    return root


def execute(args: argparse.Namespace) -> dict:
    config = read_finetune_config(DEFAULT_CONFIG)
    if config["model_name"] != NORBERTO_ID or config["max_length"] != 512:
        raise ValueError("The main entry point requires the approved NorBERTo-512 configuration.")
    args.config = DEFAULT_CONFIG
    if args.command == "train":
        args.mode = "final"
    elif args.command == "test":
        args.mode = "predict-test"
        args.output = args.output or Path("deliveries") / args.test.name
        args.output_dir = args.output_dir or Path("results/norberto/delivery") / f"prediction-{args.output.stem}"
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    with threadpool_limits(limits=config["threads"]):
        return run(args)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    return execute(args)


if __name__ == "__main__":
    main()
