# Registro de Experimentos e Resultados

Este documento consolida os parâmetros, configurações e métricas obtidas nas rodadas de avaliação de RAG e Fine-Tuning do SpringDocs Assistant.

---

## Tabela de Experimentos

| ID | Data | Descrição do Experimento | Modelo LLM | Embedding | Retrieval Mode | Top-K | Score Thresh | Latência Média (ms) | Groundedness | Status |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| EXP-000 | 2026-10-24 | Baseline de Infraestrutura | qwen2.5:3b | BAAI/bge-small-en-v1.5 | Dense | 5 | 0.35 | - | - | Configurado |

---

## Notas de Execução e Observações
- **EXP-000**: Fundação e infraestrutura estabelecidas (Fase 1 - Tarefa 1). Qdrant configurado com coleção `spring_docs_v3_4` e serviços conteinerizados.
