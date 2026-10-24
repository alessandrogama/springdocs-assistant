# SpringDocs Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC2626?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Package Manager](https://img.shields.io/badge/uv-Astral-DE5FE9?style=flat-square&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Linter](https://img.shields.io/badge/Ruff-Passed-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://astral.sh/ruff)
[![Tests](https://img.shields.io/badge/Tests-14_Passed-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

Sistema de RAG Híbrido e alinhamento comportamental (Fine-Tuning QLoRA) construído sobre a documentação oficial do **Spring Boot 3.x**. Projetado para responder a dúvidas técnicas de engenharia de software com integridade sintática de código, citação estrita de fontes canônicas e abstenção segura em perguntas fora de escopo.

---

## Por que este projeto existe?

A maioria dos assistentes de código baseados em RAG padrão sofre de três falhas graves em ambientes de produção:

1. **Fragmentação de código no chunking**: Divisores tradicionais por caracteres (`RecursiveCharacterTextSplitter`) partem classes Java ou hierarquias YAML ao meio. Isso gera alucinações de anotações inexistentes, assinaturas corrompidas e erros de compilação.
2. **Ingestão redundante e não-idempotente**: Sem identificadores determinísticos, reindexações geram pontos duplicados na base vetorial ou forçam recriações completas de índices que custam tempo computacional e chamadas repetidas de embedding.
3. **Alucinação por falta de abstenção**: Quando uma dúvida não é coberta pela documentação, LLMs comerciais tendem a inventar métodos ou misturar convenções legadas do Spring Boot 2.x com a versão 3.x.

O **SpringDocs Assistant** foi concebido para resolver esses problemas com uma arquitetura explícita, controlada e auditável.

---

## Decisões de Engenharia

- **Pipeline RAG Explícito sem LangChain/LlamaIndex**: Implementação direta utilizando clientes oficiais tipados (`qdrant-client`, `httpx`, `fastembed`, `pydantic-settings`). O pipeline é transparente, fácil de debugar e oferece telemetria real de latência de recuperação vs. geração ([ADR 001](docs/adr/001-pipeline-explicito-vs-langchain.md)).
- **Chunking Semântico com Preservação de Código**: Blocos delimitados de código (` ```java `, ` ```yaml `, AsciiDoc) são tratados como blocos atômicos indivisíveis. O overlap de 10-15% é aplicado unicamente em parágrafos de texto corrido ([ADR 002](docs/adr/002-chunking-preservando-codigo-e-idempotencia.md)).
- **Idempotência Determinística via UUIDv5 + SHA-256**: Cada chunk recebe um UUIDv5 gerado a partir de `spring_version + doc_url + section_title + chunk_index`. O cálculo de embeddings e upsert vetorial só é disparado se o `content_hash` (SHA-256) tiver mudado.
- **Busca Híbrida com Reciprocal Rank Fusion (RRF)**: O Qdrant consulta em paralelo vetores densos (`BAAI/bge-small-en-v1.5`, 384 dim, Cosine) e vetores esparsos léxicos (`Qdrant/bm25`), unificando os resultados via RRF nativo com filtro estrito de `spring_version`.
- **Grounding e Abstenção Segura**: O prompt do sistema impõe respostas em pt-BR técnico com citação formal no padrão `[Fonte: Título (URL)]` e recusa cordial padronizada caso o contexto recuperado seja insuficiente.

---

## Resultados da Fase 1 (Baseline EXP-001)

Métricas coletadas através do script de validação automatizada ([scripts/smoke_test_phase1.py](scripts/smoke_test_phase1.py)) com 10 cenários (8 factuais e 2 fora de escopo):

| Métrica | Resultado Obtido | Meta Fase 1 |
|:---|:---:|:---:|
| **Taxa de Citação Correta (Factual)** | **87.5%** (7 de 8) | ≥ 80% |
| **Taxa de Abstenção Segura (Out-of-scope)** | **100.0%** (2 de 2) | 100% |
| **Latência Média de Recuperação (Qdrant RRF)** | **82.25 ms** | < 200 ms |
| **Tempo de Reindexação Idempotente** | **0.163 s** (19 chunks ignorados) | Delta puro |
| **Testes Unitários Automatizados** | **14 / 14 aprovados** | 100% |

Os registros completos de parâmetros e latências estão disponíveis em [docs/resultados.md](docs/resultados.md).

---

## Estrutura do Repositório

```text
springdocs-assistant/
├── .agents/
│   ├── rules/                 # Regras arquiteturais e governança de personas
│   └── workflows/             # Workflows operacionais (/fase1-rag)
├── docs/
│   ├── adr/                   # Architecture Decision Records (ADRs)
│   ├── arquitetura.md         # Diagrama Mermaid e detalhamento técnico
│   └── resultados.md          # Registro experimental e histórico de benchmarks
├── scripts/
│   └── smoke_test_phase1.py   # Bateria de teste automatizada com 10 perguntas
├── src/
│   ├── ingestion/             # Parser semântico, chunker com proteção de código, hasher e indexer
│   │   ├── chunker.py
│   │   ├── curated_docs.py
│   │   ├── embedder.py
│   │   ├── hasher.py
│   │   ├── indexer.py
│   │   ├── models.py
│   │   ├── parser.py
│   │   └── pipeline.py
│   ├── rag/                   # Recuperador híbrido, construtor de prompts e cliente Ollama
│   │   ├── generator.py
│   │   ├── models.py
│   │   ├── prompt.py
│   │   ├── retriever.py
│   │   └── service.py
│   ├── config.py              # Centralização de configs via Pydantic Settings
│   └── main.py                # Aplicação FastAPI com telemetria e /ask
├── tests/                     # Suíte de testes unitários com pytest
├── docker-compose.yml         # Serviços conteinerizados: Qdrant e Ollama
├── Dockerfile                 # Multi-stage build para API em produção
└── pyproject.toml             # Metadados do projeto gerenciados via uv
```

---

## Como Executar

### 1. Pré-requisitos
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (gerenciador ultra-rápido de pacotes Python)
- Docker & Docker Compose

### 2. Subir os Serviços de Infraestrutura
Inicie o banco vetorial **Qdrant** e o runtime **Ollama**:
```bash
docker compose up -d qdrant ollama
```
- Dashboard do Qdrant: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- Endpoint da API do Ollama: [http://localhost:11434](http://localhost:11434)

Baixe o modelo LLM no container do Ollama:
```bash
docker exec springdocs-ollama ollama pull qwen2.5:3b
```

### 3. Instalar Dependências
```bash
uv sync --extra dev
```

### 4. Executar os Testes Unitários
```bash
uv run pytest tests/ -v
uv run ruff check .
```

### 5. Executar o Pipeline de Ingestão e Indexação
Processa a documentação técnica oficial, gera embeddings densos/esparsos e indexa no Qdrant:
```bash
uv run python -m src.ingestion.pipeline
```

### 6. Subir a API FastAPI
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```
- Documentação interativa (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check detalhado: [http://localhost:8000/health](http://localhost:8000/health)

### 7. Executar a Suíte de Validação (Smoke Test)
```bash
uv run python scripts/smoke_test_phase1.py
```

---

## Exemplo de Uso da API

### Requisição (`POST /ask`)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Como configurar o pool de conexões HikariCP no Spring Boot 3?"
  }'
```

### Resposta Estruturada
```json
{
  "answer": "No Spring Boot 3, o HikariCP é o pool de conexões padrão escolhido automaticamente quando `spring-boot-starter-jdbc` ou `spring-boot-starter-data-jpa` está no classpath. Você pode configurar o tamanho máximo do pool e tempos limite via `application.yaml`:\n\n```yaml\nspring:\n  datasource:\n    hikari:\n      maximum-pool-size: 20\n      minimum-idle: 5\n      connection-timeout: 30000\n```\n\n[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)]",
  "citations": [
    "[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)]"
  ],
  "retrieval_latency_ms": 84.47,
  "generation_latency_ms": 17700.01,
  "total_latency_ms": 17784.52,
  "model_name": "qwen2.5:3b",
  "spring_version": "3.4",
  "is_abstained": false
}
```

---

## Roadmap

- [x] **Fase 1 — RAG Baseline & Engenharia de Dados**:
  - Parser semântico e chunker com preservação estrita de código.
  - Ingestão idempotente via UUIDv5 e SHA-256 no Qdrant.
  - Busca híbrida (Dense + Sparse BM25 via RRF).
  - API FastAPI com telemetria por requisição e suite de smoke test.
- [ ] **Fase 2 — Dataset de Alinhamento & Fine-Tuning QLoRA**:
  - Geração de dataset sintético ChatML com isolamento anti-leakage contra `data/eval/`.
  - Treinamento QLoRA em 4-bit (Unsloth/TRL) focado em citação estrita e abstenção segura.
  - Exportação de pesos GGUF e versionamento do `Modelfile` para o Ollama.
- [ ] **Fase 3 — Avaliação Sistemática e Benchmarks Ragas**:
  - Avaliação automatizada de Groundedness, Faithfulness e Context Recall.
  - Comparação A/B entre modelo base (`qwen2.5:3b`) vs. modelo fine-tuned.
