# SpringDocs Assistant 🍃

Assistente de inteligência artificial de nível produção com **RAG Híbrido** e **Fine-Tuning (QLoRA)** treinado sobre a documentação oficial do **Spring Boot 3.x**.

---

## 🚀 Principais Características

- **RAG Explícito & de Baixa Latência**: Sem dependência de abstrações de alto nível (LangChain / LlamaIndex). Desenvolvido diretamente com os clientes oficiais `qdrant-client`, `httpx` e `Pydantic v2`.
- **Chunking Semântico com Preservação de Código**: Classes Java e configurações YAML/properties nunca são truncadas no meio, garantindo integridade sintática e de compilação.
- **Idempotência Determinística**: Geração de IDs de chunks via UUIDv5 e verificação por hash SHA-256 para upserts condicionais sem reprocessamento redundante.
- **Busca Híbrida no Qdrant**: Combinação de busca densa (vetores semânticos) e busca esparsa (BM25/Splade) com fusão Reciprocal Rank Fusion (RRF) e filtros nativos por versão do Spring Boot.
- **Alinhamento e Fine-Tuning**: Modelo local Ollama (Qwen2.5-3B) com alinhamento de comportamento para citação estrita das fontes no formato `[Fonte: Título da Seção (URL)]` e recusa segura quando a documentação não responde à pergunta.

---

## 🛠️ Stack Tecnológica

- **Linguagem & Runtime**: Python 3.11+ gerenciado via `uv`
- **Framework Web**: FastAPI com tipagem estrita Pydantic v2
- **Banco Vetorial**: Qdrant (Docker)
- **Motor de Inferência Local**: Ollama
- **Embeddings**: `fastembed` / `sentence-transformers` (BGE-small / BGE-M3)
- **Qualidade & Testes**: Ruff, Pytest

---

## 📁 Estrutura do Repositório

```text
springdocs-assistant/
├── .agents/
│   └── rules/                  # Regras e personas de desenvolvimento
├── docs/
│   ├── adr/                    # Architecture Decision Records
│   │   ├── 001-pipeline-explicito-vs-langchain.md
│   │   └── 002-chunking-preservando-codigo-e-idempotencia.md
│   ├── arquitetura.md          # Arquitetura detalhada com diagrama Mermaid
│   └── resultados.md           # Log de experimentos e métricas
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuração centralizada com Pydantic Settings
│   └── main.py                 # FastAPI app e endpoints de monitoramento
├── tests/
│   ├── __init__.py
│   └── test_config.py          # Testes automatizados de configuração
├── docker-compose.yml          # Qdrant + FastAPI api
├── Dockerfile                  # Container da API baseado em Python 3.11-slim + uv
├── pyproject.toml              # Manifesto do projeto e dependências
├── .env.example                # Variáveis de ambiente documentadas
└── .gitignore                  # Regras contra dados brutos, pesos e segredos
```

---

## ⚡ Inicialização Rápida

### 1. Pré-requisitos
- Python 3.11+ ou `uv`
- Docker & Docker Compose
- Ollama instalado localmente com o modelo:
  ```bash
  ollama pull qwen2.5:3b
  ```

### 2. Configuração do Ambiente Local
```bash
# Instalação com uv
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate no Windows

uv pip install -e ".[dev]"
```

### 3. Subir o Banco Vetorial Qdrant
```bash
docker compose up -d qdrant
```
O painel do Qdrant estará acessível em: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### 4. Executar a API Localmente
```bash
uvicorn src.main:app --reload --port 8000
```
Swagger UI disponível em: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Executar os Testes
```bash
pytest
```
