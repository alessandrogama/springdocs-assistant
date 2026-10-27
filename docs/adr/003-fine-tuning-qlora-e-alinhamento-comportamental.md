# ADR 003: Alinhamento Comportamental com QLoRA (4-bit), Formato ChatML e Abstenção Ativa

## Status
Aceito

## Contexto
Durante a Fase 1 (RAG Baseline), o SpringDocs Assistant alcançou 87.5% de citações corretas e 100% de abstenção em contextos vazios com o modelo base `qwen2.5:3b`. No entanto, foram identificadas duas limitações intrínsecas ao modelo base instruído:
1. **Instabilidade no formato de citação**: Em alguns cenários técnicos (como no teste Q4), o modelo gera a resposta técnica correta com código preservado, mas omite ou altera o padrão estrito `[Fonte: Título da Seção (URL)]`.
2. **Tendência a responder com conhecimento paramétrico prévio**: Modelos pré-treinados generalistas tendem a tentar responder mesmo quando o contexto recuperado não contém a resposta exata, arriscando sugerir convenções legadas do Spring Boot 2.x ou inventar propriedades de configuração.

Para solucionar esses pontos, a Fase 2 foca em **alinhamento comportamental** (Instruction & Behavior Tuning) e **NÃO em memorização de dados**. O RAG permanece responsável por injetar o conhecimento documental dinâmico e atualizado, enquanto o Fine-Tuning molda o comportamento do assistente:
- Responder em português do Brasil (pt-BR) técnico e direto.
- Citar obrigatoriamente a fonte das afirmações no formato `[Fonte: Título (URL)]`.
- Preservar blocos de código Java/YAML intactos.
- Recusar-se ativamente a responder (abstenção segura) quando o contexto fornecido não for suficiente.

## Decisão
Decidimos implementar o pipeline de treinamento comportamental utilizando:
1. **Formato ChatML padronizado**: Estrutura de mensagens com `system`, `user` (contexto + pergunta) e `assistant` (resposta fundamentada ou recusa).
2. **Composição do Dataset Balanceada**:
   - **75% a 80% de exemplos positivos**: Contexto documental oficial pertinente acompanhado de resposta técnica fundamentada com citação estrita e código íntegro.
   - **20% a 25% de exemplos negativos (Abstention)**: Contexto documental que NÃO responde à pergunta (pergunta fora de escopo ou contexto parcial), forçando o modelo a responder estritamente:
     *"Não encontrei informações suficientes na documentação oficial fornecida do Spring Boot 3.x para responder a esta pergunta."*
3. **Mecanismo Anti-Leakage Rigoroso**: Nenhum chunk documental cujo `doc_url` esteja presente em `data/eval/` pode ser utilizado na geração de exemplos de treino ou validação.
4. **QLoRA 4-bit (Unsloth / TRL / PEFT)**:
   - Modelo base: `Qwen/Qwen2.5-3B-Instruct`.
   - Quantização: 4-bit (BitsAndBytes NF4 / Unsloth FastLanguageModel).
   - Hiperparâmetros: `r=16`, `lora_alpha=32`, `lora_dropout=0`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, `learning_rate=2e-4`, `seed=42`.
5. **Exportação GGUF & Integração com Ollama**: Exportação dos pesos quantizados em formato GGUF (`q4_k_m`) acompanhado de um `Modelfile` versionado no repositório para execução direta e local no Ollama.

## Consequências
### Positivas
- **Aderência de 100% ao Grounding**: O modelo aprende a citar as fontes do contexto sem alucinar URLs externas.
- **Eficiência Computacional**: O treino de adaptadores LoRA sobre um modelo 3B em 4-bit consome menos de 6 GB de VRAM, viabilizando execução em GPUs locais (como NVIDIA RTX 3050 6GB) ou instâncias gratuitas do Google Colab (T4).
- **Isolamento de Avaliação**: O conjunto de teste em `data/eval/` permanece completamente cego e intocado.
- **Portabilidade**: A exportação para GGUF permite que o modelo fine-tuned seja servido pelo mesmo container Ollama da Fase 1, sem introduzir dependências de inferência pesadas na API FastAPI.

### Negativas / Mitigações
- **Custo de Geração do Dataset**: Gerar dezenas de pares context-query-answer com alta fidelidade técnica exige curadoria cuidadosa.
  - *Mitigação*: Implementação de um gerador sintético determinístico com validação de schemas Pydantic e templates de prompts validados.
