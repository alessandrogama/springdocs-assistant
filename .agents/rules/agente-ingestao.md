---
trigger: always_on
---

# Agente de Ingestão e Engenharia de Dados
Papel: Coleta, parsing semântico, chunking inteligente e indexação idempotente.

## Regras de Implementação
- Preservação de Código: NUNCA quebrar blocos de código (` ``` ` ou blocos `[source,java]`/`[source,yaml]`) no meio durante o chunking. Se uma seção com código exceder o limite, mantenha o bloco íntegro ou use estratégia Parent-Child.
- Chunking semântico por cabeçalhos/seções (alvo: 350-500 tokens, overlap de 10-15% apenas em texto corrido).
- Metadados obrigatórios em cada chunk: `doc_url`, `spring_version`, `section_title`, `module` (ex: web, data, security), `content_hash` (SHA-256) e `has_code` (bool).
- Idempotência determinística: gerar o ID de cada chunk usando UUIDv5 baseado em `(spring_version + doc_url + section_title + chunk_index)`. Reexecutar a ingestão deve atualizar apenas chunks cujo `content_hash` mudou.
- Testes obrigatórios com `pytest` cobrindo: limpeza de marcação, preservação de blocos de código Java/YAML e geração determinística de IDs.