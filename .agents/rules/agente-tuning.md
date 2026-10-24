---
trigger: always_on
---

# Agente de Fine-Tuning (QLoRA)
Papel: Geração de dataset de alinhamento comportamental, treino QLoRA e exportação GGUF.

## Regras de Implementação
- Foco do Tuning: alinhar comportamento (responder em pt-BR técnico, concisão, citação estrita das fontes fornecidas no prompt e recusa segura), NÃO memorizar fatos.
- Composição do Dataset (JSONL formato ChatML):
  - 75-80% exemplos positivos: contexto contém a resposta -> resposta concisa com citação exata e código preservado.
  - 20-25% exemplos negativos (Abstention): contexto recuperado NÃO responde à pergunta -> modelo deve responder exatamente que não encontrou na documentação fornecida.
- Anti-Leakage: verificar por `doc_url` que nenhum chunk presente em `data/eval/` seja utilizado para gerar dados de treino ou validação.
- Reprodutibilidade: fixar `seed=42`, usar Qwen2.5-3B-Instruct (ou Llama-3.2-3B-Instruct) com Unsloth/TRL em 4-bit, registrar hiperparâmetros (`r`, `lora_alpha`, `target_modules`, `lr`) e salvar o `Modelfile` do Ollama versionado no repositório.