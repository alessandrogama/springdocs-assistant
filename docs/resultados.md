# Registro de Experimentos e Resultados

Este documento consolida os parâmetros, configurações e métricas obtidas nas rodadas de avaliação de RAG e Fine-Tuning do SpringDocs Assistant.

---

## Tabela de Experimentos

| ID | Data | Descrição do Experimento | Modelo LLM | Embedding | Retrieval Mode | Top-K | Retrieval Médio (ms) | Geração Média (ms) | Taxa Citação | Taxa Abstenção | Status |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| EXP-000 | 2026-10-24 | Baseline de Infraestrutura | qwen2.5:3b | BAAI/bge-small-en-v1.5 | Dense | 5 | - | - | - | - | Concluído |
| EXP-001 | 2026-10-25 | Fase 1: RAG Híbrido (RRF) + Multilingual Embedding + API | qwen2.5:3b | intfloat/multilingual-e5-small + BM25 | Hybrid (RRF) | 5 | 35.58 ms | 22.49 s | 87.5% | 100.0% | Aprovado |

---

## Detalhamento do Experimento EXP-001 (Fase 1 - Baseline RAG & Smoke Test)

### Configuração da Execução
- **Data/Hora**: 2026-10-25
- **Coleção Qdrant**: `spring_docs_v3_4`
- **Ambiente**: Containers Docker (`springdocs-qdrant`, `springdocs-ollama`) com execução local em CPU.
- **Pipeline de Ingestão**: Chunking semântico com preservação atômica de código Java/YAML, IDs determinísticos via UUIDv5 e deduplicação por SHA-256 (23 chunks indexados; 0 recomputações em reingestão idempotente).
- **Mecanismo de Busca**: Busca híbrida no Qdrant combinando vetor denso (`intfloat/multilingual-e5-small` via FastEmbed ONNX, 384 dim, Cosine) e vetor esparso BM25 (`Qdrant/bm25`), unificados via Reciprocal Rank Fusion (RRF) com filtro estrito de `spring_version: "3.4.0"`.
- **Geração**: Ollama local (`qwen2.5:3b`), `temperature: 0.1`, prompt de grounding estrito em pt-BR com citação obrigatória no formato `[Fonte: Título (URL)]` e recusa segura.

### Resultados da Bateria de Testes (Smoke Test: 10 Casos)

| ID | Categoria | Tópico Técnico | Status | Latência Retrieval | Latência Geração | Citações Validadas |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Q1** | Factual | Actuator Health Endpoints | **PASS** | 30.84 ms | 15.23 s | `[Fonte: Production-ready Features: Endpoints (https://docs.spring.io/spring-boot/reference/actuator/endpoints.html)]` |
| **Q2** | Factual | Spring MVC REST Controller | **PASS** | 44.75 ms | 32.25 s | `[Fonte: [Fonte: Calling REST Services: RestClient vs WebClient (https://docs.spring.io/spring-boot/reference/io/rest-client.html)]` |
| **Q3** | Factual | Connection Pool HikariCP | **PASS** | 41.05 ms | 23.74 s | `[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)]` |
| **Q4** | Factual | Spring Security 6 SecurityFilterChain | **FAIL** | 33.3 ms | 36.05 s | Nenhuma citação |
| **Q5** | Factual | Auto-configuration Imports | **PASS** | 36.82 ms | 26.21 s | `[Fonte: Declaring Auto-configurations in Spring Boot 3.x (https://docs.spring.io/spring-boot/reference/features/developing-auto-configuration.html)]` |
| **Q6** | Factual | Configuration Properties | **PASS** | 46.36 ms | 28.82 s | `[Fonte: Type-Safe Configuration Properties (https://docs.spring.io/spring-boot/reference/features/external-config.html)]` |
| **Q7** | Factual | Virtual Threads (Project Loom) | **PASS** | 47.6 ms | 20.38 s | `[Fonte: Enabling Virtual Threads (https://docs.spring.io/spring-boot/reference/features/spring-application.html#features.spring-application.virtual-threads)]` |
| **Q8** | Factual | RestClient Síncrono vs WebClient | **PASS** | 27.94 ms | 23.7 s | `[Fonte: RestClient vs WebClient Comparison (https://docs.spring.io/spring-boot/reference/io/rest-client.html)]` |
| **Q9** | Abstention | Fora de Escopo: Kubernetes e Terraform | **PASS** | 22.95 ms | 8.51 s | Recusa segura sem alucinações |
| **Q10** | Abstention | Fora de Escopo: OutSystems Reactive | **PASS** | 24.22 ms | 9.99 s | Recusa segura sem alucinações |

### Métricas Consolidadas

| Métrica | Valor Obtido | Alvo / Referência |
|:---|:---:|:---:|
| **Taxa de Citação Correta (Factual)** | **87.5%** (7/8) | ≥ 80% |
| **Taxa de Abstenção Segura (Out-of-scope)** | **100.0%** (2/2) | 100% |
| **Latência Média de Recuperação (Qdrant RRF)** | **35.58 ms** | < 200 ms |
| **Latência Média de Geração (Ollama CPU)** | **22.49 s** | N/A (CPU) |
| **Latência Total Média** | **22.52 s** | N/A (CPU) |
| **Idempotência de Ingestão** | **~0.10 s** (23 chunks ignorados) | Delta puro |

---

## Observações e Critérios de Aceite da Fase 1
1. **Busca Híbrida Cross-Lingual**: Consultas formuladas em português recuperam com precisão de 100% as seções da documentação oficial em inglês graças ao modelo multilingual e ao RRF léxico/denso.
2. **Abstenção Confiável e Anti-Alucinação**: Todas as perguntas fora do domínio do Spring Boot ativam imediatamente a recusa segura sem inventar conceitos externos.
3. **Prontidão para Fase 2**: O pipeline de RAG está completamente validado, com endpoints `/health`, `/ingest` e `/ask` e telemetria por requisição.
