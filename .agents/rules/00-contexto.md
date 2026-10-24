---
trigger: always_on
---

# SpringDocs Assistant - Contexto Global
Projeto de portfólio de nível produção: RAG + Fine-tuning (QLoRA) sobre a documentação oficial do Spring Boot 3.x.

## Stack e Padrões de Código
- Python 3.11+ gerenciado via `uv` (ou Poetry), FastAPI, Pydantic v2, Qdrant, Ollama.
- Tipagem estrita obrigatória (Type Hints em todas as funções e retornos) e validação com Pydantic.
- Código, nomes de variáveis, endpoints e commits (Conventional Commits) em inglês; documentação, prompts e respostas em pt-BR.
- Simplicidade e controle: implementar o pipeline de RAG de forma explícita (clientes oficiais `qdrant-client`, `httpx`/`ollama`, `sentence-transformers`), sem abstrações opacas de frameworks como LangChain.
- Configuração centralizada via `pydantic-settings` lendo de variáveis de ambiente (`.env`).

## Governança
- Toda decisão arquitetural relevante gera um ADR enxuto em `docs/adr/`.
- Proibido commitar datasets brutos grandes, bancos locais, pesos de modelos (`.gguf`, `.safetensors`) ou segredos.
- Todo experimento deve registrar métricas, hiperparâmetros e latência em `docs/resultados.md`.