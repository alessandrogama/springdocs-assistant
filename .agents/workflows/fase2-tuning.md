# Workflow: Fase 2 — Fine-Tuning QLoRA e Alinhamento Comportamental

Este workflow descreve a esteira de execução da **Fase 2**: geração do dataset de alinhamento em ChatML com isolamento anti-leakage, treinamento QLoRA 4-bit e exportação GGUF/Modelfile para o Ollama.

---

## 1. Geração do Dataset Sintético ChatML

O script lê os chunks processados em `data/processed/chunks.jsonl`, valida o isolamento contra `data/eval/eval_set.jsonl` (descartando qualquer chunk avaliativo) e gera os arquivos balanceados em formato ChatML com divisão 80/20.

```bash
# Executa a geração determinística do dataset com 75% positivos e 25% abstenção
uv run python scripts/generate_dataset.py \
  --eval-path data/eval/eval_set.jsonl \
  --chunks-path data/processed/chunks.jsonl \
  --train-out data/train.jsonl \
  --val-out data/val.jsonl \
  --positive-ratio 0.75 \
  --split 0.80 \
  --seed 42
```

Saídas geradas:
- `data/train.jsonl`: 19 amostras em formato ChatML
- `data/val.jsonl`: 5 amostras em formato ChatML

---

## 2. Treinamento QLoRA em 4-bit

O treinamento utiliza quantização NF4 em 4-bit, adaptadores LoRA em todas as camadas lineares (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) e hiperparâmetros fixados para consumo de VRAM inferior a 6 GB.

### Em Ambiente Local ou CI (Validação Rápida / Dry-Run):
```bash
uv run python scripts/train_qlora.py --dry-run
```

### Em Ambiente com GPU Dedicada (CUDA):
```bash
uv run python scripts/train_qlora.py \
  --train-data data/train.jsonl \
  --val-data data/val.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --output-dir models/checkpoints \
  --epochs 3 \
  --lr 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed 42
```

Artefatos gerados:
- `models/checkpoints/adapter_config.json`: configuração dos pesos LoRA
- `models/checkpoints/training_summary.json`: log e métricas de perda da execução

---

## 3. Exportação GGUF e Registro no Ollama

Gera o arquivo `models/Modelfile` versionado no repositório com o template ChatML e stop tokens adequados, e exibe o guia de quantização GGUF:

```bash
uv run python scripts/export_gguf.py \
  --checkpoint-dir models/checkpoints \
  --modelfile-path models/Modelfile \
  --base-model qwen2.5:3b \
  --quantization q4_k_m
```

### Registro no Ollama Local:
```bash
ollama create springdocs-qwen3b-instruct -f models/Modelfile
```

---

## 4. Testes Automatizados e Cobertura
```bash
# Executa linter estrito
uv run ruff check src/ tests/ scripts/

# Executa suite de testes com barreira de 80% de cobertura
uv run pytest
```
