"""Export and packaging module for GGUF quantization and Ollama Modelfile."""

import logging
from pathlib import Path
from typing import Any

from src.generation.prompts import SYSTEM_PROMPT_RAG

logger = logging.getLogger(__name__)

CHATML_OLLAMA_TEMPLATE: str = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}
<|im_start|>assistant
"""


def build_modelfile_content(
    base_model_or_gguf: str = "qwen2.5:3b",
    temperature: float = 0.1,
    top_p: float = 0.9,
    num_predict: int = 1024,
    system_prompt: str | None = None,
) -> str:
    """Generate the contents of an Ollama Modelfile configured for ChatML alignment."""
    sys_prompt = system_prompt or SYSTEM_PROMPT_RAG.strip()

    modelfile = f"""# ==============================================================================
# SpringDocs Assistant - Ollama Modelfile (Aligned QLoRA / ChatML)
# ==============================================================================
FROM {base_model_or_gguf}

# Sampling parameters optimized for grounded RAG generation
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_predict {num_predict}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# ChatML conversation template
TEMPLATE \"\"\"{CHATML_OLLAMA_TEMPLATE}\"\"\"

# Aligned System Prompt with strict grounding and formal citation rules
SYSTEM \"\"\"{sys_prompt}\"\"\"
"""
    return modelfile


def export_modelfile(
    output_path: Path,
    base_model_or_gguf: str = "qwen2.5:3b",
    temperature: float = 0.1,
    top_p: float = 0.9,
    num_predict: int = 1024,
) -> Path:
    """Write the Ollama Modelfile to destination path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_modelfile_content(
        base_model_or_gguf=base_model_or_gguf,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Ollama Modelfile successfully generated at: {output_path}")
    return output_path


def get_gguf_conversion_guide(
    checkpoint_dir: Path,
    output_gguf: Path,
    quantization_type: str = "q4_k_m",
) -> dict[str, Any]:
    """Provide reproducible steps and commands for llama.cpp GGUF conversion."""
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "output_gguf": str(output_gguf),
        "quantization_type": quantization_type,
        "conversion_steps": [
            "1. Merge LoRA adapter weights with base Qwen2.5-3B model:",
            f"   python -m peft.merge_adapter --base_model_name_or_path Qwen/Qwen2.5-3B-Instruct --adapter_dir {checkpoint_dir} --output_dir models/merged",
            "2. Convert merged model to 16-bit GGUF with llama.cpp:",
            "   python llama.cpp/convert_hf_to_gguf.py models/merged/ --outfile models/exported/model-f16.gguf",
            "3. Quantize GGUF to target bit-width (e.g., q4_k_m):",
            f"   ./llama.cpp/llama-quantize models/exported/model-f16.gguf {output_gguf} {quantization_type.upper()}",
            "4. Register model in Ollama using the generated Modelfile:",
            "   ollama create springdocs-qwen3b-instruct -f models/Modelfile",
        ],
    }
