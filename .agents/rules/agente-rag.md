---
trigger: always_on
---

# Agente RAG e API
Papel: Recuperação híbrida, engenharia de prompt, saída estruturada e API resiliente.

## Regras de Implementação
- Busca Híbrida no Qdrant: combinar busca densa (embeddings multilíngues) com busca lexical/esparsa (BM25/Sparse) usando RRF (Reciprocal Rank Fusion) e suporte a filtro por `spring_version`.
- Prompt e Grounding: exigir resposta estritamente baseada nos chunks recuperados, citando no formato `[Fonte: Título da Seção (URL)]` e recusando responder quando o contexto for insuficiente.
- Todos os hiperparâmetros de recuperação (`top_k`, `score_threshold`, `rerank_enabled`, `model_name`, `temperature`) devem ser injetáveis via config/request para permitir testes A/B automatizados pelo script de avaliação.
- Telemetria por request: a resposta de `/ask` deve incluir tempos separados (`retrieval_latency_ms`, `generation_latency_ms`) e a lista de chunks recuperados com seus scores.
- API FastAPI com schemas Pydantic v2 estritos, tratamento global de exceções, timeouts configurados para o Ollama e endpoint `/health` verificando conectividade com Qdrant e Ollama.