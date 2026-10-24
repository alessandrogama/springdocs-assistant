# ADR 002: Chunking Semântico com Preservação de Código e Idempotência Determinística

## Status
Aceito

## Contexto
A documentação técnica do Spring Boot 3.x é composta por explicações conceituais entremeadas por blocos densos de código Java (`@Configuration`, `@Bean`, anotações do Spring Data/Security) e arquivos de configuração estruturados (YAML e `application.properties`).

No processamento clássico de RAG:
1. **Quebra de Sintaxe e Semântica de Código**: Estratégias convencionais de chunking baseadas puramente em contagem de caracteres (`RecursiveCharacterTextSplitter`) rotineiramente partem classes Java ou hierarquias YAML ao meio. Isso resulta em alucinações severas pelo LLM, snippets incompletos e erros de compilação quando o usuário tenta executar o código fornecido.
2. **Reingestão Não-Idempotente**: Atualizações na documentação ou reexecuções do script de ingestão frequentemente recriam IDs aleatórios (UUIDv4) para os mesmos trechos, duplicando pontos na base vetorial ou exigindo um `delete_collection` oneroso que apaga dados inalterados.

## Decisão
Adotamos uma estratégia de engenharia de dados baseada em dois pilares:

### 1. Chunking Semântico com Preservação de Código
- **Segmentação por Cabeçalhos e Seções**: O particionamento primário segue os limites semânticos da documentação (títulos de seção H2/H3 e blocos lógicos), mantendo um tamanho alvo de **350 a 500 tokens** com overlap de 10% a 15% aplicado **apenas em texto corrido**.
- **Preservação Rígida de Blocos de Código**: Blocos delimitados por ` ``` `, `[source,java]` ou `[source,yaml]` são tratados como **unidades indivisíveis**. Caso um bloco de código exceda o limite nominal do chunk, ele é mantido íntegro ou associado a uma hierarquia de chunks *Parent-Child*, garantindo que nenhuma instrução Java ou chave YAML seja truncada.
- **Metadados Obrigatórios**: Todo chunk inclui os metadados:
  - `doc_url`: URL oficial canônica do trecho.
  - `spring_version`: Versão do Spring Boot indexada (ex: `3.4`).
  - `section_title`: Nome da seção/cabeçalho pai.
  - `module`: Módulo Spring correlato (ex: `web`, `data`, `security`, `actuator`).
  - `content_hash`: Hash SHA-256 do conteúdo bruto normalizado.
  - `has_code`: Booleano identificando presença de trechos executáveis.

### 2. Idempotência Determinística
- **Geração de IDs via UUIDv5**: O ID do ponto no Qdrant é calculado deterministicamente usando UUIDv5 a partir de um namespace fixo e da chave composta:
  $$\text{Key} = \text{spring\_version} + \text{doc\_url} + \text{section\_title} + \text{chunk\_index}$$
- **Atualização Condicional por `content_hash`**: Durante reingestões, o hash SHA-256 do chunk gerado é comparado com o hash já presente no Qdrant para aquele UUID. Apenas chunks novos ou cujo conteúdo tenha sido alterado acionam o cálculo de embeddings e upsert vetorial, economizando tempo computacional e chamadas de modelo.

## Alternativas Consideradas
1. **Divisão Cega por Caracteres / Tokens (Fixed-size Sliding Window)**:
   - *Prós*: Implementação trivial.
   - *Contras*: Degrada drasticamente o contexto de código, separando anotações de suas classes ou truncando estruturas YAML aninhadas.
2. **UUIDv4 Aleatório com Recriação da Coleção (Drop & Recreate)**:
   - *Prós*: Sem necessidade de gerenciar hashes ou IDs determinísticos.
   - *Contras*: Causa indisponibilidade durante reindexações, consome recursos computacionais refazendo embeddings de 100% dos dados inalterados e impede atualizações incrementais em produção.
3. **Chunking Exclusivo por Classe/Arquivo Java**:
   - *Prós*: Preserva código em arquivos individuais.
   - *Contras*: Não se aplica diretamente a manuais de referência e guias que misturam explicações textuais com snippets parciais.

## Consequências
### Positivas
- **Fidelidade Sintática Total**: O LLM recebe trechos de código Java e configurações YAML com sua sintaxe intacta, reduzindo alucinações a zero na geração de snippets.
- **Idempotência Real e Economia de Recursos**: Reexecutar a ingestão é uma operação segura e rápida, processando apenas deltas de documentação.
- **Rastreabilidade e Filtros Estritos**: IDs previsíveis facilitam auditorias, testes unitários automatizados e remoção seletiva de chunks obsoletos.

### Negativas / Mitigações
- **Variação no Tamanho dos Chunks**: Chunks com blocos de código grandes podem ultrapassar ligeiramente a janela alvo de 500 tokens.
  - *Mitigação*: Janela de contexto do modelo de embedding (como BGE-small / BGE-M3 de 512 a 8192 tokens) e do LLM (Qwen2.5 de 32k tokens) suportam tranquilamente blocos atômicos maiores de código sem truncamento.
