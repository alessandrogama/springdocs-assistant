"""Data models and schemas for ChatML dataset generation and QLoRA fine-tuning."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """Single message in ChatML format."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, description="Message content in ChatML turn")


class ChatMLSample(BaseModel):
    """Complete multi-turn conversational sample in ChatML format."""

    messages: list[ChatMessage] = Field(
        ...,
        min_length=2,
        description="List of messages representing system, user, and assistant turns",
    )

    @model_validator(mode="after")
    def validate_chatml_structure(self) -> "ChatMLSample":
        """Validate that the last turn is assistant and contains at least one user query."""
        roles = [m.role for m in self.messages]
        if "user" not in roles:
            raise ValueError("ChatML sample must contain at least one 'user' message")
        if self.messages[-1].role != "assistant":
            raise ValueError("Last message in ChatML sample must be from 'assistant'")
        return self


class DatasetConfig(BaseModel):
    """Configuration for alignment dataset generation."""

    eval_path: Path = Field(
        default=Path("data/eval/eval_set.jsonl"),
        description="Path to evaluation set for anti-leakage isolation",
    )
    processed_chunks_path: Path = Field(
        default=Path("data/processed/chunks.jsonl"),
        description="Path to processed documentary chunks",
    )
    train_output_path: Path = Field(
        default=Path("data/train.jsonl"),
        description="Output path for training dataset in ChatML JSONL",
    )
    val_output_path: Path = Field(
        default=Path("data/val.jsonl"),
        description="Output path for validation dataset in ChatML JSONL",
    )
    positive_ratio: float = Field(
        default=0.75,
        ge=0.70,
        le=0.85,
        description="Target proportion of positive grounded samples (0.75-0.80 recommended)",
    )
    train_val_split: float = Field(
        default=0.80,
        ge=0.50,
        le=0.95,
        description="Proportion of samples allocated to training set",
    )
    seed: int = Field(default=42, description="Random seed for deterministic dataset shuffling")


class DatasetStats(BaseModel):
    """Execution statistics for dataset generation."""

    total_samples: int
    train_samples: int
    val_samples: int
    positive_samples: int
    negative_samples: int
    positive_ratio: float
    negative_ratio: float
    leaked_chunks_rejected: int
    anti_leakage_verified: bool


class TuningHyperparameters(BaseModel):
    """Hyperparameters for 4-bit QLoRA fine-tuning."""

    base_model_name: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Base instruct model identifier from Hugging Face",
    )
    r: int = Field(default=16, description="LoRA attention rank dimension")
    lora_alpha: int = Field(default=32, description="LoRA scaling parameter")
    lora_dropout: float = Field(default=0.0, description="LoRA dropout rate (0.0 for Unsloth)")
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target linear projection modules for adapter injection",
    )
    learning_rate: float = Field(default=2e-4, description="Optimizer initial learning rate")
    per_device_train_batch_size: int = Field(
        default=1,
        description="Micro-batch size per GPU device (optimized for 6GB VRAM)",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Number of gradient accumulation steps",
    )
    num_train_epochs: int = Field(default=3, description="Number of full training epochs")
    max_seq_length: int = Field(default=1024, description="Maximum token sequence length")
    seed: int = Field(default=42, description="Random seed for training reproducibility")
    use_4bit: bool = Field(default=True, description="Enable 4-bit quantization (BitsAndBytes NF4)")
    output_dir: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory to save adapter weights and training logs",
    )
    dry_run: bool = Field(
        default=False,
        description="When True, validates pipeline without executing GPU-heavy training",
    )
