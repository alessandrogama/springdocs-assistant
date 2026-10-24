---
description: 
---

# Workflow: /fase1-rag (Execução e Validação da Fase 1)

1. Verifique se os serviços de infraestrutura estão ativos (`docker compose ps`) e valide a conectividade com Qdrant e Ollama.
2. Execute a suíte de testes unitários de ingestão e chunking (`pytest tests/ -v`).
3. Execute o pipeline de ingestão e indexação para o escopo da Fase 1 (`python -m src.ingestion.pipeline`).
4. Suba ou valide a API FastAPI e teste o endpoint `GET /health`.
5. Execute o script `python scripts/smoke_test_phase1.py` contendo as 10 perguntas de validação (8 factuais sobre Spring Boot 3.x e 2 perguntas fora de escopo para testar abstenção).
6. Gere o relatório inicial em `docs/resultados.md` com taxa de citação correta, taxa de abstenção e latência média (`retrieval_ms` e `generation_ms`).