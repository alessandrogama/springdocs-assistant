"""CLI script to run 4-bit QLoRA fine-tuning for SpringDocs Assistant."""

import argparse
import logging
import sys
from pathlib import Path

from src.tuning.models import TuningHyperparameters
from src.tuning.trainer import QLoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_qlora")


def main() -> None:
    """Parse hyperparameters and execute training."""
    parser = argparse.ArgumentParser(
        description="Run 4-bit QLoRA fine-tuning for ChatML instruction alignment"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/val.jsonl"),
        help="Path to val.jsonl",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model from Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Output directory for checkpoints and adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation and simulation without training on GPU",
    )

    args = parser.parse_args()

    hyperparams = TuningHyperparameters(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    trainer = QLoRATrainer(hyperparams)

    try:
        summary = trainer.train(train_path=args.train_data, val_path=args.val_data)
        print("\n=== QLoRA Training Completed Successfully ===")
        print(f"Status:         {summary.get('status')}")
        print(f"Execution Mode: {summary.get('mode')}")
        print(f"Base Model:     {summary.get('base_model')}")
        print(f"Train Samples:  {summary.get('train_samples')}")
        print(f"Val Samples:    {summary.get('val_samples')}")
        print(f"Adapter Output: {summary.get('adapter_path')}")
        if "final_train_loss" in summary:
            print(f"Final Train Loss: {summary.get('final_train_loss')}")
            print(f"Final Eval Loss:  {summary.get('final_eval_loss')}")
        print("==============================================\n")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
