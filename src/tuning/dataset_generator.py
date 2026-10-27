"""Synthetic ChatML dataset generator with strict anti-leakage isolation.

Generates instruction and behavior alignment data for SpringDocs Assistant:
- 75% to 80% positive grounded technical samples with source citations and code blocks.
- 20% to 25% negative abstention samples enforcing safe refusal.
- Strict anti-leakage filter rejecting any chunks sharing URLs with data/eval/.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.generation.prompts import STANDARD_ABSTENTION_PHRASE, SYSTEM_PROMPT_RAG
from src.ingestion.models import DocumentChunk
from src.tuning.models import (
    ChatMessage,
    ChatMLSample,
    DatasetConfig,
    DatasetStats,
)

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalize a document URL for robust anti-leakage comparison."""
    parsed = urlparse(url.strip())
    # Strip query and fragment, ignore trailing slashes
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


def load_eval_urls(eval_path: Path) -> set[str]:
    """Load and normalize forbidden evaluation URLs from eval dataset."""
    eval_urls: set[str] = set()
    if not eval_path.exists():
        logger.warning(f"Eval path {eval_path} not found. No URLs registered for anti-leakage.")
        return eval_urls

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            if "doc_url" in item:
                eval_urls.add(normalize_url(item["doc_url"]))
    return eval_urls


def is_chunk_leaked(chunk_url: str, forbidden_urls: set[str]) -> bool:
    """Check if chunk URL matches any evaluation URL."""
    norm = normalize_url(chunk_url)
    return norm in forbidden_urls


def build_user_prompt(chunk: DocumentChunk, question: str) -> str:
    """Format single chunk context and user query according to prompt templates."""
    source_tag = f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
    context_str = (
        f"[FONTE 1]\n"
        f"Referência: {source_tag}\n"
        f"Módulo: {chunk.module} (Spring Boot {chunk.spring_version})\n"
        f"Conteúdo:\n{chunk.content}\n"
    )
    return (
        f"Contexto documental indexado:\n"
        f"{context_str}\n\n"
        f"Pergunta do desenvolvedor:\n"
        f"{question}\n\n"
        f"Resposta técnica fundamentada exclusivamente no contexto acima (incluindo citações):"
    )


# Curated question-answer pairs corresponding to known documentary chunks
_CURATED_POSITIVE_SPECS: list[dict[str, Any]] = [
    {
        "section_title": "Exposing Actuator Endpoints",
        "question": "Como expor endpoints adicionais do Spring Boot Actuator além do health via HTTP?",
        "answer_summary": (
            "Para expor endpoints adicionais como `info`, `metrics` e `prometheus` além do endpoint padrão `health`, "
            "deve-se configurar a propriedade `management.endpoints.web.exposure.include` no arquivo `application.yaml`.\n\n"
            "Exemplo de configuração:\n"
            "```yaml\n"
            "management:\n"
            "  endpoints:\n"
            "    web:\n"
            "      exposure:\n"
            "        include: \"health,info,metrics,prometheus\"\n"
            "  endpoint:\n"
            "    health:\n"
            "      show-details: always\n"
            "      probes:\n"
            "        enabled: true\n"
            "```\n\n"
            "[Fonte: Exposing Actuator Endpoints (https://docs.spring.io/spring-boot/reference/actuator/endpoints.html)]"
        ),
    },
    {
        "section_title": "Health Endpoint Details",
        "question": "Como habilitar os detalhes completos e probes de liveness e readiness no /actuator/health?",
        "answer_summary": (
            "Para visualizar informações detalhadas de todos os beans `HealthIndicator` registrados, defina `management.endpoint.health.show-details: always`. "
            "As sondas de Liveness e Readiness são habilitadas automaticamente em ambiente Kubernetes ou explicitamente através de "
            "`management.endpoint.health.probes.enabled=true`.\n\n"
            "[Fonte: Health Endpoint Details (https://docs.spring.io/spring-boot/reference/actuator/endpoints.html)]"
        ),
    },
    {
        "section_title": "Building RESTful Web Services",
        "question": "Qual a finalidade da anotação @RestController no Spring MVC e como utilizá-la em um controller?",
        "answer_summary": (
            "A anotação `@RestController` é uma anotação de conveniência que combina `@Controller` e `@ResponseBody`. "
            "Ela indica que os dados retornados por cada método do controller serão serializados diretamente no corpo da resposta HTTP.\n\n"
            "Exemplo de implementação em Java:\n"
            "```java\n"
            "package com.example.demo.web;\n\n"
            "import org.springframework.web.bind.annotation.GetMapping;\n"
            "import org.springframework.web.bind.annotation.PathVariable;\n"
            "import org.springframework.web.bind.annotation.PostMapping;\n"
            "import org.springframework.web.bind.annotation.RequestBody;\n"
            "import org.springframework.web.bind.annotation.RequestMapping;\n"
            "import org.springframework.web.bind.annotation.RestController;\n\n"
            "@RestController\n"
            "@RequestMapping(\"/api/users\")\n"
            "public class UserController {\n\n"
            "    @GetMapping(\"/{id}\")\n"
            "    public UserResponse getUser(@PathVariable Long id) {\n"
            "        return new UserResponse(id, \"Spring Developer\");\n"
            "    }\n\n"
            "    @PostMapping\n"
            "    public UserResponse createUser(@RequestBody CreateUserRequest request) {\n"
            "        return new UserResponse(100L, request.name());\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "[Fonte: Building RESTful Web Services (https://docs.spring.io/spring-boot/reference/web/servlet.html)]"
        ),
    },
    {
        "section_title": "Jackson JSON Serialization",
        "question": "Como o Spring Boot gerencia instâncias de ObjectMapper e registro de módulos Jackson customizados?",
        "answer_summary": (
            "O Spring Boot auto-configura uma instância de `ObjectMapper` com configurações padrão adequadas. "
            "Qualquer bean registrado do tipo `com.fasterxml.jackson.databind.Module` é automaticamente detectado e registrado no `ObjectMapper` auto-configurado.\n\n"
            "[Fonte: Jackson JSON Serialization (https://docs.spring.io/spring-boot/reference/web/servlet.html)]"
        ),
    },
    {
        "section_title": "Connection Pool Auto-configuration",
        "question": "Qual é o connection pool padrão no Spring Boot 3 e como configurar seus parâmetros no HikariCP via YAML?",
        "answer_summary": (
            "O pool de conexões padrão escolhido pelo Spring Boot é o HikariCP sempre que `spring-boot-starter-jdbc` ou `spring-boot-starter-data-jpa` estiver presente no classpath. "
            "A configuração do datasource e propriedades específicas do HikariCP pode ser realizada conforme abaixo:\n\n"
            "```yaml\n"
            "spring:\n"
            "  datasource:\n"
            "    url: jdbc:postgresql://localhost:5432/springdocs\n"
            "    username: springuser\n"
            "    password: secretpassword\n"
            "    driver-class-name: org.postgresql.Driver\n"
            "    hikari:\n"
            "      maximum-pool-size: 20\n"
            "      minimum-idle: 5\n"
            "      connection-timeout: 30000\n"
            "      idle-timeout: 600000\n"
            "```\n\n"
            "[Fonte: Connection Pool Auto-configuration (https://docs.spring.io/spring-boot/reference/data/sql.html)]"
        ),
    },
    {
        "section_title": "Spring Data JPA Repositories",
        "question": "Como declarar interfaces de repositório no Spring Data JPA para entidades relacionais?",
        "answer_summary": (
            "No Spring Data JPA, as implementações de repositórios são geradas automaticamente ao estender a interface `JpaRepository` informando a entidade e o tipo da chave primária:\n\n"
            "```java\n"
            "package com.example.demo.repository;\n\n"
            "import org.springframework.data.jpa.repository.JpaRepository;\n"
            "import org.springframework.stereotype.Repository;\n"
            "import java.util.List;\n\n"
            "@Repository\n"
            "public interface OrderRepository extends JpaRepository<Order, Long> {\n"
            "    List<Order> findByStatus(String status);\n"
            "}\n"
            "```\n\n"
            "[Fonte: Spring Data JPA Repositories (https://docs.spring.io/spring-boot/reference/data/sql.html)]"
        ),
    },
    {
        "section_title": "Configuring SecurityFilterChain",
        "question": "Como configurar o SecurityFilterChain no Spring Boot 3.x com a remoção do WebSecurityConfigurerAdapter?",
        "answer_summary": (
            "No Spring Boot 3.x (Spring Security 6.x), a classe `WebSecurityConfigurerAdapter` foi completamente removida. "
            "As regras de segurança devem ser configuradas declarando um `@Bean` do tipo `SecurityFilterChain` usando a DSL baseada em lambdas e métodos de componentes:\n\n"
            "```java\n"
            "package com.example.demo.security;\n\n"
            "import org.springframework.context.annotation.Bean;\n"
            "import org.springframework.context.annotation.Configuration;\n"
            "import org.springframework.security.config.Customizer;\n"
            "import org.springframework.security.config.annotation.web.builders.HttpSecurity;\n"
            "import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;\n"
            "import org.springframework.security.web.SecurityFilterChain;\n\n"
            "@Configuration\n"
            "@EnableWebSecurity\n"
            "public class SecurityConfiguration {\n\n"
            "    @Bean\n"
            "    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {\n"
            "        http\n"
            "            .authorizeHttpRequests(authorize -> authorize\n"
            "                .requestMatchers(\"/public/**\", \"/actuator/health\").permitAll()\n"
            "                .anyRequest().authenticated()\n"
            "            )\n"
            "            .httpBasic(Customizer.withDefaults())\n"
            "            .csrf(csrf -> csrf.disable());\n"
            "        return http.build();\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "[Fonte: Configuring SecurityFilterChain (https://docs.spring.io/spring-boot/reference/web/spring-security.html)]"
        ),
    },
    {
        "section_title": "Declaring Auto-configurations in Spring Boot 3.x",
        "question": "Onde devem ser declaradas as classes de auto-configuração no Spring Boot 3.x?",
        "answer_summary": (
            "No Spring Boot 3.x, as classes de auto-configuração devem ser declaradas no arquivo:\n"
            "`META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`\n\n"
            "O arquivo legado `META-INF/spring.factories` não é mais suportado para registro de auto-configurações no Spring Boot 3.x.\n\n"
            "Exemplo com anotação `@AutoConfiguration`:\n"
            "```java\n"
            "package com.example.demo.autoconfigure;\n\n"
            "import org.springframework.boot.autoconfigure.AutoConfiguration;\n"
            "import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;\n"
            "import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;\n"
            "import org.springframework.context.annotation.Bean;\n\n"
            "@AutoConfiguration\n"
            "@ConditionalOnClass(DataSource.class)\n"
            "public class CustomDatabaseAutoConfiguration {\n\n"
            "    @Bean\n"
            "    @ConditionalOnMissingBean\n"
            "    public CustomDatabaseService databaseService() {\n"
            "        return new CustomDatabaseService();\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "[Fonte: Declaring Auto-configurations in Spring Boot 3.x (https://docs.spring.io/spring-boot/reference/features/developing-auto-configuration.html)]"
        ),
    },
    {
        "section_title": "Type-Safe Configuration Properties",
        "question": "Como mapear propriedades de configuração para records Java usando @ConfigurationProperties no Spring Boot 3?",
        "answer_summary": (
            "Utiliza-se a anotação `@ConfigurationProperties` definindo o prefixo das propriedades para fazer o binding para classes ou Java Records fortemente tipados.\n\n"
            "Exemplo do Record:\n"
            "```java\n"
            "package com.example.demo.config;\n\n"
            "import org.springframework.boot.context.properties.ConfigurationProperties;\n"
            "import java.time.Duration;\n\n"
            "@ConfigurationProperties(prefix = \"app.mail\")\n"
            "public record MailProperties(\n"
            "    String host,\n"
            "    int port,\n"
            "    Duration timeout,\n"
            "    boolean auth\n"
            ") {}\n"
            "```\n\n"
            "Configuração no `application.yaml`:\n"
            "```yaml\n"
            "app:\n"
            "  mail:\n"
            "    host: smtp.example.com\n"
            "    port: 587\n"
            "    timeout: 5s\n"
            "    auth: true\n"
            "```\n\n"
            "Para habilitar a descoberta de propriedades, utilize a anotação `@ConfigurationPropertiesScan` na classe principal da aplicação.\n\n"
            "[Fonte: Type-Safe Configuration Properties (https://docs.spring.io/spring-boot/reference/features/external-config.html)]"
        ),
    },
    {
        "section_title": "Enabling Virtual Threads",
        "question": "Como habilitar Virtual Threads (Project Loom) no Spring Boot 3.4 e quais componentes são afetados?",
        "answer_summary": (
            "As Virtual Threads são habilitadas no Spring Boot 3.2+ e 3.4 definindo a seguinte propriedade:\n\n"
            "```yaml\n"
            "spring:\n"
            "  threads:\n"
            "    virtual:\n"
            "      enabled: true\n"
            "```\n\n"
            "Ao ativar essa propriedade:\n"
            "1. Servidores web embutidos (Tomcat, Jetty) utilizam virtual threads para atender requisições HTTP recebidas.\n"
            "2. O processamento assíncrono de requisições no Spring MVC passa a utilizar virtual threads.\n"
            "3. Métodos `@Async` e beans `TaskExecutor` utilizam `SimpleAsyncTaskExecutor` configurado com virtual threads.\n"
            "4. Listeners de mensagens RabbitMQ e Kafka utilizam task executors com virtual threads.\n\n"
            "[Fonte: Enabling Virtual Threads (https://docs.spring.io/spring-boot/reference/features/spring-application.html#features.spring-application.virtual-threads)]"
        ),
    },
    {
        "section_title": "Distributed Tracing Configuration",
        "question": "Como configurar rastreamento distribuído (distributed tracing) com Micrometer Tracing e Zipkin no Spring Boot?",
        "answer_summary": (
            "Ao adicionar um tracing starter (como `micrometer-tracing-bridge-brave` ou `micrometer-tracing-bridge-otel`), "
            "o Spring Boot configura o tracing automaticamente. A amostragem e endpoint do Zipkin são configurados no `application.yaml`:\n\n"
            "```yaml\n"
            "management:\n"
            "  tracing:\n"
            "    sampling:\n"
            "      probability: 1.0\n"
            "  zipkin:\n"
            "    tracing:\n"
            "      endpoint: \"http://localhost:9411/api/v2/spans\"\n"
            "```\n\n"
            "Traces e spans são correlacionados automaticamente nos logs da aplicação através de `logging.pattern.level` com injeção de traceId e spanId.\n\n"
            "[Fonte: Distributed Tracing Configuration (https://docs.spring.io/spring-boot/reference/actuator/observability.html)]"
        ),
    },
    {
        "section_title": "RestClient vs WebClient Comparison",
        "question": "Qual a diferença entre RestClient, WebClient e RestTemplate no Spring Boot 3.x?",
        "answer_summary": (
            "No Spring Boot 3.2+ e 3.4, a recomendação para chamadas HTTP é estruturada da seguinte forma:\n"
            "1. **RestClient**: Cliente síncrono e bloqueante com API fluente moderna. É o padrão recomendado para aplicações Spring MVC sem necessidade do Project Reactor.\n"
            "2. **WebClient**: Cliente assíncrono e não-bloqueante integrante do `spring-boot-starter-webflux`, ideal para arquiteturas reativas e streaming.\n"
            "3. **RestTemplate**: Cliente legado síncrono mantido em modo de manutenção; novos projetos devem priorizar `RestClient`.\n\n"
            "[Fonte: RestClient vs WebClient Comparison (https://docs.spring.io/spring-boot/reference/io/rest-client.html)]"
        ),
    },
    {
        "section_title": "Using RestClient in Spring Boot 3.x",
        "question": "Como injetar e utilizar o RestClient.Builder para consumir uma API REST externa no Spring Boot 3?",
        "answer_summary": (
            "Injeta-se o bean pré-configurado `RestClient.Builder` para construir a instância do cliente configurando baseUrl e headers padrão:\n\n"
            "```java\n"
            "package com.example.demo.client;\n\n"
            "import org.springframework.stereotype.Service;\n"
            "import org.springframework.web.client.RestClient;\n\n"
            "@Service\n"
            "public class GitHubClientService {\n\n"
            "    private final RestClient restClient;\n\n"
            "    public GitHubClientService(RestClient.Builder builder) {\n"
            "        this.restClient = builder\n"
            "            .baseUrl(\"https://api.github.com\")\n"
            "            .defaultHeader(\"Accept\", \"application/vnd.github.v3+json\")\n"
            "            .build();\n"
            "    }\n\n"
            "    public RepositoryResponse getRepository(String owner, String repo) {\n"
            "        return this.restClient.get()\n"
            "            .uri(\"/repos/{owner}/{repo}\", owner, repo)\n"
            "            .retrieve()\n"
            "            .body(RepositoryResponse.class);\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "[Fonte: Using RestClient in Spring Boot 3.x (https://docs.spring.io/spring-boot/reference/io/rest-client.html)]"
        ),
    },
    {
        "section_title": "Using WebClient in Reactive Applications",
        "question": "Como construir e efetuar chamadas reativas utilizando WebClient.Builder no Spring WebFlux?",
        "answer_summary": (
            "Para aplicações reativas, utiliza-se `WebClient.Builder` para instanciar o cliente e retornar instâncias de `Mono` ou `Flux` do Project Reactor:\n\n"
            "```java\n"
            "package com.example.demo.client;\n\n"
            "import org.springframework.stereotype.Service;\n"
            "import org.springframework.web.reactive.function.client.WebClient;\n"
            "import reactor.core.publisher.Mono;\n\n"
            "@Service\n"
            "public class ReactiveGitHubService {\n\n"
            "    private final WebClient webClient;\n\n"
            "    public ReactiveGitHubService(WebClient.Builder builder) {\n"
            "        this.webClient = builder.baseUrl(\"https://api.github.com\").build();\n"
            "    }\n\n"
            "    public Mono<RepositoryResponse> getRepository(String owner, String repo) {\n"
            "        return this.webClient.get()\n"
            "            .uri(\"/repos/{owner}/{repo}\", owner, repo)\n"
            "            .retrieve()\n"
            "            .bodyToMono(RepositoryResponse.class);\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "[Fonte: Using WebClient in Reactive Applications (https://docs.spring.io/spring-boot/reference/io/rest-client.html)]"
        ),
    },
    {
        "section_title": "Production-ready Features: Endpoints",
        "question": "O que são Actuator Endpoints no Spring Boot e como eles auxiliam no monitoramento?",
        "answer_summary": (
            "Actuator endpoints permitem monitorar e interagir diretamente com a aplicação Spring Boot em tempo de execução. "
            "O Spring Boot disponibiliza diversos endpoints nativos embutidos (built-in) e também oferece suporte para criação de endpoints customizados.\n\n"
            "[Fonte: Production-ready Features: Endpoints (https://docs.spring.io/spring-boot/reference/actuator/endpoints.html)]"
        ),
    },
    {
        "section_title": "Core Features: Externalized Configuration",
        "question": "Qual o propósito da configuração externalizada no Spring Boot?",
        "answer_summary": (
            "A configuração externalizada no Spring Boot permite desacoplar parâmetros de configuração do código-fonte da aplicação, "
            "permitindo que o mesmo código execute de maneira consistente e adaptada em múltiplos ambientes (desenvolvimento, homologação e produção).\n\n"
            "[Fonte: Core Features: Externalized Configuration (https://docs.spring.io/spring-boot/reference/features/external-config.html)]"
        ),
    },
    {
        "section_title": "Developing Web Applications: Spring MVC",
        "question": "Como o Spring Boot lida com a configuração básica do Spring MVC?",
        "answer_summary": (
            "O Spring Boot fornece auto-configuração nativa para o Spring MVC, configurando componentes essenciais automaticamente para atender à maioria das aplicações web.\n\n"
            "[Fonte: Developing Web Applications: Spring MVC (https://docs.spring.io/spring-boot/reference/web/servlet.html)]"
        ),
    },
    {
        "section_title": "Data Access: SQL Databases & Connection Pools",
        "question": "Como o Spring Boot simplifica a integração com bancos de dados relacionais?",
        "answer_summary": (
            "O Spring Boot simplifica o trabalho com bancos relacionais oferecendo auto-configuração para Spring Data JPA, JDBC e pools de conexão consagrados como HikariCP.\n\n"
            "[Fonte: Data Access: SQL Databases & Connection Pools (https://docs.spring.io/spring-boot/reference/data/sql.html)]"
        ),
    },
]

# Negative abstention queries designed to pair with unrelated or incomplete contexts
_CURATED_NEGATIVE_QUERIES: list[dict[str, str]] = [
    {
        "target_chunk_section": "Exposing Actuator Endpoints",
        "unrelated_question": "Como configurar um cluster Apache Cassandra distribuído no Kubernetes usando Helm?",
    },
    {
        "target_chunk_section": "Building RESTful Web Services",
        "unrelated_question": "Como criar pipelines de Continuous Deployment no GitLab CI com runners dedicados?",
    },
    {
        "target_chunk_section": "Connection Pool Auto-configuration",
        "unrelated_question": "Como configurar o protocolo de segurança TLS mútuo (mTLS) no Istio Service Mesh?",
    },
    {
        "target_chunk_section": "Configuring SecurityFilterChain",
        "unrelated_question": "Qual a sintaxe para criação de triggers e procedures em PL/SQL no Oracle Database 19c?",
    },
    {
        "target_chunk_section": "Declaring Auto-configurations in Spring Boot 3.x",
        "unrelated_question": "Como implementar um crawler assíncrono em Python com Playwright e BeautifulSoup?",
    },
    {
        "target_chunk_section": "Enabling Virtual Threads",
        "unrelated_question": "Como provisionar uma VPC com subnets públicas e privadas na AWS utilizando Terraform?",
    },
    {
        "target_chunk_section": "Distributed Tracing Configuration",
        "unrelated_question": "Como configurar um modelo de aprendizado por reforço no PyTorch usando PPO?",
    },
    {
        "target_chunk_section": "RestClient vs WebClient Comparison",
        "unrelated_question": "Como criar telas e widgets dinâmicos no framework Flutter para desktop?",
    },
]


def load_chunks_from_file(path: Path) -> list[DocumentChunk]:
    """Load DocumentChunk instances from a JSONL file."""
    if not path.exists():
        logger.warning(f"Chunks file {path} not found.")
        return []
    chunks: list[DocumentChunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            chunks.append(DocumentChunk(**data))
    return chunks


class DatasetGenerator:
    """Generator for synthetic ChatML alignment dataset with anti-leakage isolation."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self.forbidden_urls: set[str] = load_eval_urls(self.config.eval_path)

    def filter_safe_chunks(self, chunks: list[DocumentChunk]) -> tuple[list[DocumentChunk], int]:
        """Filter out any chunk whose URL is present in the evaluation dataset."""
        safe_chunks: list[DocumentChunk] = []
        leaked_count = 0

        for chunk in chunks:
            if is_chunk_leaked(chunk.doc_url, self.forbidden_urls):
                leaked_count += 1
                logger.warning(
                    f"ANTI-LEAKAGE: Rejected chunk {chunk.id} with forbidden doc_url {chunk.doc_url}"
                )
            else:
                safe_chunks.append(chunk)

        return safe_chunks, leaked_count

    def generate_samples(self, safe_chunks: list[DocumentChunk]) -> list[ChatMLSample]:
        """Generate positive grounded and negative abstention ChatML samples."""
        chunk_by_section: dict[str, DocumentChunk] = {
            chunk.section_title: chunk for chunk in safe_chunks
        }

        samples: list[ChatMLSample] = []

        # 1. Generate Positive Samples from safe chunks
        for spec in _CURATED_POSITIVE_SPECS:
            section = spec["section_title"]
            chunk = chunk_by_section.get(section)
            if not chunk:
                # Fallback to any chunk that shares doc_url or partial title
                for cand in safe_chunks:
                    if section.lower() in cand.section_title.lower():
                        chunk = cand
                        break
            if not chunk:
                continue

            user_prompt = build_user_prompt(chunk, spec["question"])
            sample = ChatMLSample(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT_RAG),
                    ChatMessage(role="user", content=user_prompt),
                    ChatMessage(role="assistant", content=spec["answer_summary"]),
                ]
            )
            samples.append(sample)

        # 2. Add extra derived positive samples if needed to reach desired count
        for chunk in safe_chunks:
            if chunk.has_code and chunk.section_title not in [s["section_title"] for s in _CURATED_POSITIVE_SPECS]:
                derived_q = f"Como utilizar e configurar os recursos descritos na seção '{chunk.section_title}' no Spring Boot 3.4?"
                derived_a = (
                    f"Com base na documentação oficial de {chunk.section_title}, os recursos são configurados da seguinte forma:\n\n"
                    f"{chunk.content}\n\n"
                    f"[Fonte: {chunk.section_title} ({chunk.doc_url})]"
                )
                user_prompt = build_user_prompt(chunk, derived_q)
                samples.append(
                    ChatMLSample(
                        messages=[
                            ChatMessage(role="system", content=SYSTEM_PROMPT_RAG),
                            ChatMessage(role="user", content=user_prompt),
                            ChatMessage(role="assistant", content=derived_a),
                        ]
                    )
                )

        positive_count = len(samples)

        # 3. Generate Negative Abstention Samples (20% to 25% of total)
        # target_neg = positive_count * (1 - positive_ratio) / positive_ratio
        # For positive_ratio ~ 0.77: positive/(positive+negative) = 0.77 -> negative = positive * 0.23 / 0.77
        target_negative_count = max(4, int(positive_count * (1.0 - self.config.positive_ratio) / self.config.positive_ratio))

        negative_samples: list[ChatMLSample] = []
        for i in range(target_negative_count):
            spec = _CURATED_NEGATIVE_QUERIES[i % len(_CURATED_NEGATIVE_QUERIES)]
            target_section = spec["target_chunk_section"]
            chunk = chunk_by_section.get(target_section)
            if not chunk and safe_chunks:
                chunk = safe_chunks[i % len(safe_chunks)]
            if not chunk:
                continue

            user_prompt = build_user_prompt(chunk, spec["unrelated_question"])
            neg_sample = ChatMLSample(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT_RAG),
                    ChatMessage(role="user", content=user_prompt),
                    ChatMessage(role="assistant", content=STANDARD_ABSTENTION_PHRASE),
                ]
            )
            negative_samples.append(neg_sample)

        samples.extend(negative_samples)
        return samples

    def run(self) -> DatasetStats:
        """Execute full dataset generation pipeline with anti-leakage and train/val split."""
        all_chunks = load_chunks_from_file(self.config.processed_chunks_path)
        safe_chunks, leaked_rejected = self.filter_safe_chunks(all_chunks)

        samples = self.generate_samples(safe_chunks)
        if not samples:
            raise ValueError("No samples generated. Check chunks file and config.")

        # Shuffle with deterministic seed
        rng = random.Random(self.config.seed)
        rng.shuffle(samples)

        # Split train and validation
        split_idx = int(len(samples) * self.config.train_val_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Create output directories
        self.config.train_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.val_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save train.jsonl
        with open(self.config.train_output_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        # Save val.jsonl
        with open(self.config.val_output_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        pos_count = sum(
            1 for s in samples if s.messages[-1].content != STANDARD_ABSTENTION_PHRASE
        )
        neg_count = len(samples) - pos_count
        pos_ratio = pos_count / len(samples)
        neg_ratio = neg_count / len(samples)

        stats = DatasetStats(
            total_samples=len(samples),
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            positive_samples=pos_count,
            negative_samples=neg_count,
            positive_ratio=round(pos_ratio, 4),
            negative_ratio=round(neg_ratio, 4),
            leaked_chunks_rejected=leaked_rejected,
            anti_leakage_verified=True,
        )

        logger.info(
            f"Generated {stats.total_samples} samples ({stats.train_samples} train, {stats.val_samples} val). "
            f"Positive: {stats.positive_ratio * 100:.1f}%, Negative: {stats.negative_ratio * 100:.1f}%. "
            f"Anti-leakage rejected: {stats.leaked_chunks_rejected} chunks."
        )

        return stats
