# Registro de Experimentos e Resultados

Este documento consolida os parâmetros, configurações e métricas obtidas nas rodadas de avaliação de RAG e Fine-Tuning do SpringDocs Assistant.

---

## Tabela de Experimentos

| ID | Data | Descrição do Experimento | Modelo LLM | Embedding | Retrieval Mode | Top-K | Retrieval Médio (ms) | Geração Média (ms) | Taxa Citação | Taxa Abstenção | Status |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| EXP-000 | 2026-10-24 | Baseline de Infraestrutura | qwen2.5:3b | BAAI/bge-small-en-v1.5 | Dense | 5 | - | - | - | - | Concluído |
| EXP-001 | 2026-10-24 | Fase 1: RAG Híbrido (RRF) + Ingestão Idempotente | qwen2.5:3b | BAAI/bge-small + BM25 | Hybrid (RRF) | 3 | 82.25 ms | 35.85 s | 87.5% (7/8) | 100.0% (2/2) | Validado |

---

## Detalhamento do Experimento EXP-001 (Fase 1 - Baseline RAG)

### Configuração da Execução
- **Data/Hora**: 2026-10-24
- **Coleção Qdrant**: `spring_docs_v3_4`
- **Ambiente**: Containers Docker (`springdocs-qdrant`, `springdocs-ollama`) com execução local em CPU.
- **Pipeline de Ingestão**: Chunking semântico com preservação atômica de código Java/YAML, IDs determinísticos via UUIDv5 e deduplicação por SHA-256 (19 chunks indexados; 0 recomputações em reingestão idempotente).
- **Mecanismo de Busca**: Busca híbrida no Qdrant combinando vetor denso (`BAAI/bge-small-en-v1.5`, 384 dim, Cosine) e vetor esparso BM25 (`Qdrant/bm25`), unificados via Reciprocal Rank Fusion (RRF) com filtro estrito de `spring_version: "3.4"`.
- **Geração**: Ollama local (`qwen2.5:3b`), `temperature: 0.1`, `num_predict: 384`, prompt de grounding estrito em pt-BR com citação obrigatória no formato `[Fonte: Título (URL)]`.

### Resultados da Bateria de Testes (Smoke Test: 10 Casos)

| ID | Categoria | Tópico Técnico | Status | Latência Retrieval | Latência Geração | Citações Validadas |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Q1** | Factual | Actuator Health (`/actuator/health`, `show-details`) | PASS | 57.75 ms | 55.17 s | `[Fonte: Production-ready Features: Endpoints]` |
| **Q2** | Factual | Spring MVC REST Controller (`@RestController`, Jackson) | PASS | 202.31 ms | 48.49 s | `[Fonte: Developing Web Applications: Spring MVC]` |
| **Q3** | Factual | Connection Pool HikariCP (`maximum-pool-size`) | PASS | 84.47 ms | 17.70 s | `[Fonte: Connection Pool Auto-configuration]` |
| **Q4** | Factual | Spring Security 6 (`SecurityFilterChain`, descontinuação de `WebSecurityConfigurerAdapter`) | PASS | 60.88 ms | 38.63 s | `[Fonte: Configuring SecurityFilterChain]` |
| **Q5** | Factual | Auto-configuration Imports (`AutoConfiguration.imports`) | PASS | 62.91 ms | 42.78 s | `[Fonte: Declaring Auto-configurations in Spring Boot 3.x]` |
| **Q6** | Factual | Configuration Properties (`@ConfigurationProperties`) | PASS | 62.70 ms | 47.07 s | Resposta correta com código Java e YAML |
| **Q7** | Factual | Virtual Threads (`spring.threads.virtual.enabled`) | PASS | 91.51 ms | 44.28 s | `[Fonte: Enabling Virtual Threads]` |
| **Q8** | Factual | Distributed Tracing (Micrometer Tracing & sampling) | PASS | 35.17 ms | 40.82 s | `[Fonte: Production-ready Features: Observability & Tracing]` |
| **Q9** | Abstenção | Out-of-scope: Kubernetes Istio & Helm | PASS | 34.14 ms | 11.05 s | Recusa segura sem alucinações |
| **Q10** | Abstenção | Out-of-scope: React Redux Toolkit | PASS | 130.69 ms | 12.51 s | Recusa segura sem alucinações |

### Métricas Consolidadas

| Métrica | Valor Obtido | Alvo / Referência |
|:---|:---:|:---:|
| **Taxa de Citação Correta (Factual)** | **87.5%** (7/8) | ≥ 80% |
| **Taxa de Abstenção Segura (Out-of-scope)** | **100.0%** (2/2) | 100% |
| **Latência Média de Recuperação (Qdrant RRF)** | **82.25 ms** | < 200 ms |
| **Latência Média de Geração (Ollama CPU)** | **35.85 s** | N/A (CPU) |
| **Latência Total Média** | **35.93 s** | N/A (CPU) |
| **Idempotência de Ingestão** | **0.163 s** (19 chunks ignorados) | Delta puro |

---

## Observações e Próximos Passos (Fase 2)
1. **Recuperação Híbrida**: O uso de RRF nativo no Qdrant combinando vetores densos e esparsos BM25 apresentou latência média de apenas **82.25 ms**, capturando perfeitamente termos léxicos técnicos (`maximum-pool-size`, `SecurityFilterChain`, `AutoConfiguration.imports`).
2. **Abstenção Confiável**: Ambas as perguntas fora de escopo (Kubernetes/Istio e React/Redux) resultaram em 100% de abstenção segura imediata, impedindo qualquer alucinação de fatos externos.
3. **Alinhamento na Fase 2**: O modelo de fine-tuning QLoRA (Fase 2) abordará especificamente a padronização das citações (alcançando 100% de adesão estrita ao formato de citação) e redução do tempo de resposta.
