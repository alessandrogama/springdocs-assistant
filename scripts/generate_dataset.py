"""CLI script to generate synthetic ChatML dataset for fine-tuning."""

import argparse
import logging
import sys
from pathlib import Path

from src.tuning.dataset_generator import DatasetGenerator
from src.tuning.models import DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_dataset")


def main() -> None:
    """Parse arguments and execute dataset generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic ChatML dataset with anti-leakage isolation"
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("data/eval/eval_set.jsonl"),
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to processed chunks JSONL file",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Output path for train.jsonl",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("data/val.jsonl"),
        help="Output path for val.jsonl",
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.75,
        help="Target positive grounded ratio (default: 0.75)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.80,
        help="Train/val split ratio (default: 0.80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    config = DatasetConfig(
        eval_path=args.eval_path,
        processed_chunks_path=args.chunks_path,
        train_output_path=args.train_out,
        val_output_path=args.val_out,
        positive_ratio=args.positive_ratio,
        train_val_split=args.split,
        seed=args.seed,
    )

    generator = DatasetGenerator(config)
    try:
        stats = generator.run()
        print("\n=== Dataset Generation Summary ===")
        print(f"Total Samples Generated: {stats.total_samples}")
        print(f"Training Set:            {stats.train_samples} samples ({args.train_out})")
        print(f"Validation Set:          {stats.val_samples} samples ({args.val_out})")
        print(f"Positive Grounded:       {stats.positive_samples} ({stats.positive_ratio * 100:.1f}%)")
        print(f"Negative Abstention:     {stats.negative_samples} ({stats.negative_ratio * 100:.1f}%)")
        print(f"Anti-leakage Rejected:   {stats.leaked_chunks_rejected} chunks")
        print(f"Anti-leakage Verified:   {stats.anti_leakage_verified}")
        print("==================================\n")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
