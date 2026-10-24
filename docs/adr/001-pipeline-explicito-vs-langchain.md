# ADR 001: Pipeline RAG Explícito com Clientes Nativos vs. LangChain / LlamaIndex

## Status
Aceito

## Contexto
O **SpringDocs Assistant** é um projeto de nível de produção focado em RAG e Fine-Tuning sobre a documentação oficial do Spring Boot 3.x. O objetivo central é fornecer um assistente técnico de alta precisão, latência previsível e comportamento rigorosamente auditável.

Frameworks de alto nível como **LangChain** e **LlamaIndex** oferecem prototipagem rápida através de abstrações densas (como `RetrievalQA`, cadeias pré-formatadas, callbacks implícitos e geradores de prompt opacos). No entanto, em cenários de engenharia de produção, essas abstrações impõem problemas críticos:
1. **Dificuldade de depuração (debugging)**: Erros de execução geram stack traces profundos e de difícil rastreamento entre dezenas de camadas internas do framework.
2. **Falta de controle de latência**: Chamadas assíncronas ocultas, re-embeddings ou sanitizações automatizadas introduzem sobrecarga de milissegundos não controlada.
3. **Instabilidade de APIs e dependências infladas**: Atualizações frequentes quebram contratos internos, exigindo migrações contínuas e trazendo dezenas de dependências transitórias desnecessárias.
4. **Telemetria e controle fino opacos**: Requisitos como decomposição precisa de tempos (`retrieval_latency_ms` vs. `generation_latency_ms`) e rankeamento customizado (RRF para busca híbrida) tornam-se trabalhosos quando mediados por interfaces padronizadas e inflexíveis.

## Decisão
Decidimos implementar todo o pipeline RAG de forma **explícita e nativa**, sem o uso de LangChain, LlamaIndex ou bibliotecas de orquestração similares, utilizando exclusivamente:
- **`qdrant-client`**: Cliente oficial e tipado para comunicação direta via REST/gRPC com o Qdrant, permitindo controle direto sobre filtros de payload (`spring_version`), busca vetorial densa, esparsa e fusão com RRF.
- **`httpx`**: Cliente HTTP assíncrono moderno e eficiente para comunicação direta com a API do Ollama (endpoints `/api/generate` ou `/api/chat`), com timeouts explícitos por requisição e streaming nativo.
- **`pydantic v2` e `pydantic-settings`**: Validação estrita de contratos de entrada/saída, parsing estruturado e injeção de parâmetros de configuração.
- **`fastembed` / `sentence-transformers`**: Geração controlada de embeddings locais sem pipelines intermediários.

## Alternativas Consideradas
1. **LangChain (Python)**:
   - *Prós*: Ecossistema abrangente e grande acervo de tutoriais.
   - *Contras*: Excesso de camadas de abstração ("leaky abstractions"), dependências pesadas, overhead na telemetria e constantes `DeprecationWarnings`.
2. **LlamaIndex**:
   - *Prós*: Foco bem estruturado em indexação de documentos e nós de dados.
   - *Contras*: Formato de nós proprietário que dificulta o mapeamento direto e transparente para coleções nativas do Qdrant.
3. **Haystack 2.x**:
   - *Prós*: Arquitetura de pipeline explícito em grafo.
   - *Contras*: Adiciona curva de aprendizado própria de nós/componentes sem ganhos significativos em relação a um pipeline Python assíncrono nativo em FastAPI.

## Consequências
### Positivas
- **Controle Total e Previsibilidade**: Cada linha de código de recuperação e inferência é direta, legível e determinística.
- **Telemetria de Alta Precisão**: Métricas de tempo de retrieval, tempo de inferência e tokens gerados são mensuradas diretamente no escopo da requisição.
- **Dependências Enxutas**: O ambiente virtual e a imagem Docker permanecem leves, inicializam rapidamente e possuem superfície de vulnerabilidades drasticamente reduzida.
- **Facilidade em Testes Unitários**: Funções simples que recebem clientes tipados são fáceis de mockar com `pytest` sem necessidade de gerenciar contextos complexos de agentes.

### Negativas / Mitigações
- **Boilerplate de Conexão e Retry**: Precisamos implementar manualmente estratégias de retry, fallbacks e formatação de prompts.
  - *Mitigação*: Implementação de módulos utilitários coesos em `src/services/` com tratamento global de exceções no FastAPI.
