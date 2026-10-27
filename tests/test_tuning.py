"""Unit and integration tests for Phase 2 fine-tuning, ChatML dataset generation and Ollama Modelfile."""

import json
from pathlib import Path

import pytest

from src.generation.prompts import STANDARD_ABSTENTION_PHRASE
from src.ingestion.models import DocumentChunk
from src.tuning.dataset_generator import (
    DatasetGenerator,
    is_chunk_leaked,
    load_chunks_from_file,
    load_eval_urls,
    normalize_url,
)
from src.tuning.exporter import (
    build_modelfile_content,
    export_modelfile,
    get_gguf_conversion_guide,
)
from src.tuning.models import (
    ChatMessage,
    ChatMLSample,
    DatasetConfig,
    DatasetStats,
    TuningHyperparameters,
)
from src.tuning.trainer import QLoRATrainer


def test_chatml_models_validation() -> None:
    """Validate ChatML Pydantic models and structure constraints."""
    # Valid ChatML sample
    valid = ChatMLSample(
        messages=[
            ChatMessage(role="system", content="System instruction"),
            ChatMessage(role="user", content="User question"),
            ChatMessage(role="assistant", content="Assistant answer"),
        ]
    )
    assert len(valid.messages) == 3
    assert valid.messages[-1].role == "assistant"

    # Missing user turn should fail
    with pytest.raises(ValueError, match="at least one 'user' message"):
        ChatMLSample(
            messages=[
                ChatMessage(role="system", content="System instruction"),
                ChatMessage(role="assistant", content="Assistant answer"),
            ]
        )

    # Last message not assistant should fail
    with pytest.raises(ValueError, match="must be from 'assistant'"):
        ChatMLSample(
            messages=[
                ChatMessage(role="system", content="System instruction"),
                ChatMessage(role="user", content="User question"),
            ]
        )


def test_url_normalization() -> None:
    """Test URL normalization for robust anti-leakage checks."""
    url1 = "https://docs.spring.io/spring-boot/reference/actuator/metrics.html"
    url2 = "HTTPS://DOCS.SPRING.IO/spring-boot/reference/actuator/metrics.html/"
    url3 = "https://docs.spring.io/spring-boot/reference/actuator/metrics.html#section-1"

    assert normalize_url(url1) == "https://docs.spring.io/spring-boot/reference/actuator/metrics.html"
    assert normalize_url(url2) == "https://docs.spring.io/spring-boot/reference/actuator/metrics.html"
    assert normalize_url(url3) == "https://docs.spring.io/spring-boot/reference/actuator/metrics.html"


def test_anti_leakage_filter_logic(tmp_path: Path) -> None:
    """Test that chunks sharing URLs with the evaluation set are strictly rejected."""
    eval_file = tmp_path / "eval_set.jsonl"
    eval_content = [
        {"id": "EVAL-001", "doc_url": "https://docs.spring.io/spring-boot/reference/actuator/metrics.html"},
        {"id": "EVAL-002", "doc_url": "https://docs.spring.io/spring-boot/reference/web/webflux.html"},
    ]
    with open(eval_file, "w", encoding="utf-8") as f:
        for item in eval_content:
            f.write(json.dumps(item) + "\n")

    forbidden = load_eval_urls(eval_file)
    assert len(forbidden) == 2
    assert "https://docs.spring.io/spring-boot/reference/actuator/metrics.html" in forbidden

    assert is_chunk_leaked("https://docs.spring.io/spring-boot/reference/actuator/metrics.html", forbidden)
    assert is_chunk_leaked("https://docs.spring.io/spring-boot/reference/actuator/metrics.html#gauges", forbidden)
    assert not is_chunk_leaked("https://docs.spring.io/spring-boot/reference/data/sql.html", forbidden)


def test_dataset_generator_anti_leakage_rejection(tmp_path: Path) -> None:
    """Verify that the dataset generator rejects leaked chunks during execution."""
    eval_file = tmp_path / "eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"doc_url": "https://docs.spring.io/forbidden.html"}) + "\n")

    chunks = [
        DocumentChunk(
            id="chunk-safe-1",
            doc_url="https://docs.spring.io/safe1.html",
            spring_version="3.4.0",
            section_title="Exposing Actuator Endpoints",
            module="actuator",
            content="Actuator exposure docs...",
            content_hash="hash1",
            has_code=False,
            chunk_index=0,
            parent_section=None,
        ),
        DocumentChunk(
            id="chunk-leaked-1",
            doc_url="https://docs.spring.io/forbidden.html",
            spring_version="3.4.0",
            section_title="Leaked Section",
            module="security",
            content="Leaked content...",
            content_hash="hash2",
            has_code=False,
            chunk_index=0,
            parent_section=None,
        ),
    ]

    generator = DatasetGenerator(
        DatasetConfig(
            eval_path=eval_file,
            processed_chunks_path=tmp_path / "chunks.jsonl",
            train_output_path=tmp_path / "train.jsonl",
            val_output_path=tmp_path / "val.jsonl",
        )
    )

    safe, leaked = generator.filter_safe_chunks(chunks)
    assert len(safe) == 1
    assert leaked == 1
    assert safe[0].id == "chunk-safe-1"


def test_dataset_generation_proportions_and_content(tmp_path: Path) -> None:
    """Verify generated dataset proportions (75% positive, 25% negative) and grounding citations."""
    eval_file = tmp_path / "eval_set.jsonl"
    eval_file.write_text(
        json.dumps({
            "id": "EVAL-001",
            "doc_url": "https://docs.spring.io/forbidden-section.html",
            "section_title": "Forbidden Section",
        })
        + "\n",
        encoding="utf-8",
    )

    chunks_file = tmp_path / "chunks.jsonl"
    sample_titles = [
        "Exposing Actuator Endpoints",
        "Health Endpoint Details",
        "Building RESTful Web Services",
        "Jackson JSON Serialization",
        "Connection Pool Auto-configuration",
        "Spring Data JPA Repositories",
        "Configuring SecurityFilterChain",
        "Declaring Auto-configurations in Spring Boot 3.x",
        "Type-Safe Configuration Properties",
        "Enabling Virtual Threads",
        "Distributed Tracing Configuration",
        "RestClient vs WebClient Comparison",
    ]
    chunks = [
        DocumentChunk(
            id=f"chunk-{i}",
            doc_url=f"https://docs.spring.io/spring-boot/reference/section-{i}.html",
            spring_version="3.4.0",
            section_title=title,
            module="core",
            content=f"Documentation content for {title}.\n\n```java\nclass Demo{i} {{}}\n```",
            content_hash=f"hash-{i}",
            has_code=True,
            chunk_index=0,
            parent_section=title,
        )
        for i, title in enumerate(sample_titles)
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    config = DatasetConfig(
        eval_path=eval_file,
        processed_chunks_path=chunks_file,
        train_output_path=tmp_path / "train.jsonl",
        val_output_path=tmp_path / "val.jsonl",
        positive_ratio=0.75,
        train_val_split=0.80,
        seed=42,
    )

    generator = DatasetGenerator(config)
    stats: DatasetStats = generator.run()

    assert stats.total_samples > 0
    assert stats.train_samples > 0
    assert stats.val_samples > 0
    assert stats.train_samples + stats.val_samples == stats.total_samples
    assert 0.70 <= stats.positive_ratio <= 0.80
    assert 0.20 <= stats.negative_ratio <= 0.30
    assert stats.anti_leakage_verified is True

    # Validate output files exist and contain valid ChatML records
    assert config.train_output_path.exists()
    assert config.val_output_path.exists()

    with open(config.train_output_path, encoding="utf-8") as f:
        train_records = [ChatMLSample(**json.loads(line)) for line in f]
    assert len(train_records) == stats.train_samples

    # Verify that positive records have source citations and negative records have standard abstention
    pos_found = 0
    neg_found = 0
    for record in train_records:
        assistant_content = record.messages[-1].content
        if assistant_content == STANDARD_ABSTENTION_PHRASE:
            neg_found += 1
        else:
            pos_found += 1
            assert "[Fonte:" in assistant_content
            assert "(" in assistant_content and ")" in assistant_content

    assert pos_found > 0
    assert neg_found > 0


def test_qlora_hyperparameters_defaults() -> None:
    """Verify fine-tuning hyperparameters match ADR 003 specifications."""
    params = TuningHyperparameters()
    assert params.base_model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert params.r == 16
    assert params.lora_alpha == 32
    assert params.learning_rate == 2e-4
    assert params.num_train_epochs == 3
    assert params.use_4bit is True
    assert params.seed == 42
    assert "q_proj" in params.target_modules
    assert "k_proj" in params.target_modules
    assert "v_proj" in params.target_modules


def test_qlora_trainer_simulation_and_artifacts(tmp_path: Path) -> None:
    """Test trainer execution, validation, and adapter generation in dry-run mode."""
    # Create mock dataset
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    sample = ChatMLSample(
        messages=[
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="User query"),
            ChatMessage(role="assistant", content="Answer with [Fonte: Title (https://spring.io)]"),
        ]
    )

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample.model_dump()) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample.model_dump()) + "\n")

    output_dir = tmp_path / "checkpoints"
    hyperparams = TuningHyperparameters(
        output_dir=output_dir,
        dry_run=True,
    )

    trainer = QLoRATrainer(hyperparams)
    summary = trainer.train(train_file, val_file)

    assert summary["status"] == "completed"
    assert summary["mode"] == "dry_run"
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "training_summary.json").exists()

    with open(output_dir / "adapter_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["quantization"] == "4bit-nf4"


def test_qlora_trainer_raises_on_missing_or_empty_dataset(tmp_path: Path) -> None:
    """Verify trainer fails cleanly when dataset path is invalid or empty."""
    trainer = QLoRATrainer()
    missing_file = tmp_path / "missing.jsonl"
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()

    with pytest.raises(FileNotFoundError):
        trainer.train(missing_file, missing_file)

    with pytest.raises(ValueError, match="no valid ChatML samples"):
        trainer.train(empty_file, empty_file)


def test_qlora_trainer_raises_on_malformed_json_dataset(tmp_path: Path) -> None:
    """Verify trainer fails when dataset has invalid JSON."""
    trainer = QLoRATrainer()
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("NOT_JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ChatML record"):
        trainer.train(bad_file, bad_file)


def test_dataset_generator_edge_cases(tmp_path: Path) -> None:
    """Test missing eval file, missing chunks file, and items without doc_url."""
    # Non-existent eval path
    urls = load_eval_urls(tmp_path / "does_not_exist.jsonl")
    assert urls == set()

    # Eval file with items missing doc_url
    eval_file = tmp_path / "empty_doc_url.jsonl"
    eval_file.write_text(json.dumps({"id": "NO_URL"}) + "\n\n", encoding="utf-8")
    urls = load_eval_urls(eval_file)
    assert urls == set()

    # Non-existent chunks file
    chunks = load_chunks_from_file(tmp_path / "nonexistent.jsonl")
    assert chunks == []

    # Generator raising on empty chunks
    gen = DatasetGenerator(
        DatasetConfig(
            eval_path=eval_file,
            processed_chunks_path=tmp_path / "nonexistent.jsonl",
            train_output_path=tmp_path / "t.jsonl",
            val_output_path=tmp_path / "v.jsonl",
        )
    )
    with pytest.raises(ValueError, match="No samples generated"):
        gen.run()


def test_qlora_trainer_cuda_mocking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify cuda availability reporting with mocked torch module."""
    import sys
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3050"
    mock_props = MagicMock()
    mock_props.total_memory = 6 * (1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props

    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    trainer = QLoRATrainer()
    info = trainer.check_cuda_availability()
    assert info["torch_available"] is True
    assert info["cuda_available"] is True
    assert info["device_name"] == "NVIDIA GeForce RTX 3050"
    assert info["vram_gb"] == 6.0


def test_ollama_modelfile_and_gguf_export(tmp_path: Path) -> None:
    """Test Ollama Modelfile generation and GGUF conversion instructions."""
    content = build_modelfile_content(
        base_model_or_gguf="qwen2.5:3b",
        temperature=0.1,
    )
    assert "FROM qwen2.5:3b" in content
    assert "PARAMETER temperature 0.1" in content
    assert "PARAMETER stop \"<|im_end|>\"" in content
    assert "TEMPLATE" in content
    assert "SYSTEM" in content
    assert STANDARD_ABSTENTION_PHRASE in content

    modelfile_path = tmp_path / "Modelfile"
    exported = export_modelfile(modelfile_path, base_model_or_gguf="qwen2.5:3b")
    assert exported.exists()
    assert exported.read_text(encoding="utf-8") == content

    guide = get_gguf_conversion_guide(
        checkpoint_dir=Path("models/checkpoints"),
        output_gguf=Path("models/exported/model.gguf"),
        quantization_type="q4_k_m",
    )
    assert "conversion_steps" in guide
    assert any("llama-quantize" in step for step in guide["conversion_steps"])

