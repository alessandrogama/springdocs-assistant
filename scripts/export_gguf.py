"""CLI script to export Ollama Modelfile and provide GGUF conversion instructions."""

import argparse
import logging
import sys
from pathlib import Path

from src.tuning.exporter import export_modelfile, get_gguf_conversion_guide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_gguf")


def main() -> None:
    """Parse CLI arguments and generate Ollama Modelfile and GGUF guide."""
    parser = argparse.ArgumentParser(
        description="Export Ollama Modelfile and generate GGUF quantization workflow"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--modelfile-path",
        type=Path,
        default=Path("models/Modelfile"),
        help="Destination path for Ollama Modelfile",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="qwen2.5:3b",
        help="Base model tag in Ollama or GGUF binary path",
    )
    parser.add_argument(
        "--gguf-out",
        type=Path,
        default=Path("models/exported/springdocs-qwen3b-q4_k_m.gguf"),
        help="Target GGUF file path",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Target quantization type (q4_k_m, q5_k_m, q8_0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for Modelfile (default: 0.1)",
    )

    args = parser.parse_args()

    try:
        # 1. Export versioned Modelfile
        saved_modelfile = export_modelfile(
            output_path=args.modelfile_path,
            base_model_or_gguf=args.base_model,
            temperature=args.temperature,
        )

        # 2. Get conversion pipeline instructions
        guide = get_gguf_conversion_guide(
            checkpoint_dir=args.checkpoint_dir,
            output_gguf=args.gguf_out,
            quantization_type=args.quantization,
        )

        print("\n=== Ollama Modelfile & GGUF Export Summary ===")
        print(f"Modelfile Written To:  {saved_modelfile}")
        print(f"Target GGUF Binary:    {args.gguf_out}")
        print(f"Quantization Type:     {args.quantization.upper()}")
        print("\nReproducible llama.cpp conversion workflow:")
        for step in guide["conversion_steps"]:
            print(f"  {step}")
        print("\nTo load model directly into local Ollama:")
        print(f"  ollama create springdocs-qwen3b-instruct -f {saved_modelfile}")
        print("==============================================\n")

    except Exception as e:
        logger.error(f"Modelfile export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
