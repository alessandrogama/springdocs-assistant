"""Curated official Spring Boot 3.4 documentation dataset for Phase 1 ingestion."""

from src.ingestion.models import RawDocument

SPRING_BOOT_3_4_DOCS: list[RawDocument] = [
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/actuator/endpoints.html",
        spring_version="3.4.0",
        module="actuator",
        title="Production-ready Features: Endpoints",
        content="""# Production-ready Features: Endpoints

Actuator endpoints let you monitor and interact with your application. Spring Boot includes a number of built-in endpoints and lets you add your own.

## Exposing Actuator Endpoints
By default, only the health endpoint is exposed over HTTP. To expose other endpoints such as info, metrics, or env, configure `management.endpoints.web.exposure.include`.

```yaml
management:
  endpoints:
    web:
      exposure:
        include: "health,info,metrics,prometheus"
  endpoint:
    health:
      show-details: always
      probes:
        enabled: true
```

## Health Endpoint Details
The `/actuator/health` endpoint provides basic application health information. When `management.endpoint.health.show-details` is set to `always`, detailed information from all registered `HealthIndicator` beans is displayed, including database connectivity, disk space, and custom health checks.

Liveness and Readiness probes are automatically configured when running in a Kubernetes environment or when `management.endpoint.health.probes.enabled=true` is set.
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/web/servlet.html",
        spring_version="3.4.0",
        module="web",
        title="Developing Web Applications: Spring MVC",
        content="""# Developing Web Applications: Spring MVC

Spring Boot provides auto-configuration for Spring MVC that works well with most applications.

## Building RESTful Web Services
The `@RestController` annotation is a convenience annotation that combines `@Controller` and `@ResponseBody`. It indicates that data returned by each method will be written directly into the response body.

```java
package com.example.demo.web;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @GetMapping("/{id}")
    public UserResponse getUser(@PathVariable Long id) {
        return new UserResponse(id, "Spring Developer");
    }

    @PostMapping
    public UserResponse createUser(@RequestBody CreateUserRequest request) {
        return new UserResponse(100L, request.name());
    }
}
```

## Jackson JSON Serialization
Spring Boot auto-configures an `ObjectMapper` instance with sensible defaults. Any beans of type `com.fasterxml.jackson.databind.Module` are automatically registered with the `ObjectMapper`.
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/data/sql.html",
        spring_version="3.4.0",
        module="data",
        title="Data Access: SQL Databases & Connection Pools",
        content="""# Data Access: SQL Databases & Connection Pools

Spring Boot makes it easy to work with relational databases using Spring Data JPA, JDBC, and popular connection pools.

## Connection Pool Auto-configuration
HikariCP is the default connection pool chosen by Spring Boot when `spring-boot-starter-jdbc` or `spring-boot-starter-data-jpa` is present on the classpath.

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5432/springdocs
    username: springuser
    password: secretpassword
    driver-class-name: org.postgresql.Driver
    hikari:
      maximum-pool-size: 20
      minimum-idle: 5
      connection-timeout: 30000
      idle-timeout: 600000
```

## Spring Data JPA Repositories
Spring Data JPA creates implementations of repository interfaces automatically.

```java
package com.example.demo.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface OrderRepository extends JpaRepository<Order, Long> {
    List<Order> findByStatus(String status);
}
```
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/web/spring-security.html",
        spring_version="3.4.0",
        module="security",
        title="Security: Spring Security Configuration",
        content="""# Security: Spring Security Configuration

Spring Boot provides auto-configuration for Spring Security 6.x.

## Configuring SecurityFilterChain
In Spring Boot 3.x, `WebSecurityConfigurerAdapter` is completely removed. Security rules must be defined by declaring a `@Bean` of type `SecurityFilterChain` using component-based configuration and lambda DSL.

```java
package com.example.demo.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfiguration {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(authorize -> authorize
                .requestMatchers("/public/**", "/actuator/health").permitAll()
                .anyRequest().authenticated()
            )
            .httpBasic(Customizer.withDefaults())
            .csrf(csrf -> csrf.disable());
        return http.build();
    }
}
```
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/features/developing-auto-configuration.html",
        spring_version="3.4.0",
        module="core",
        title="Core Features: Auto-configuration",
        content="""# Core Features: Auto-configuration

Spring Boot auto-configuration attempts to automatically configure your Spring application based on the jar dependencies that you have added.

## Declaring Auto-configurations in Spring Boot 3.x
In Spring Boot 3.x, auto-configuration classes must be declared in:
`META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`

The legacy file `META-INF/spring.factories` is no longer supported for auto-configuration registration in Spring Boot 3.x.

```java
package com.example.demo.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;

@AutoConfiguration
@ConditionalOnClass(DataSource.class)
public class CustomDatabaseAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public CustomDatabaseService databaseService() {
        return new CustomDatabaseService();
    }
}
```
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/features/external-config.html",
        spring_version="3.4.0",
        module="core",
        title="Core Features: Externalized Configuration",
        content="""# Core Features: Externalized Configuration

Spring Boot lets you externalize your configuration so that you can work with the same application code in different environments.

## Type-Safe Configuration Properties
Use the `@ConfigurationProperties` annotation to bind properties from `application.properties` or `application.yaml` to strongly-typed Java records or classes.

```java
package com.example.demo.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import java.time.Duration;

@ConfigurationProperties(prefix = "app.mail")
public record MailProperties(
    String host,
    int port,
    Duration timeout,
    boolean auth
) {}
```

In `application.yaml`:
```yaml
app:
  mail:
    host: smtp.example.com
    port: 587
    timeout: 5s
    auth: true
```
Enable binding using `@ConfigurationPropertiesScan` on your `@SpringBootApplication` class.
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/features/spring-application.html#features.spring-application.virtual-threads",
        spring_version="3.4.0",
        module="core",
        title="Core Features: Virtual Threads",
        content="""# Core Features: Virtual Threads

Spring Boot 3.2+ and 3.4 provide first-class support for Virtual Threads (Project Loom) available in Java 21+.

## Enabling Virtual Threads
Virtual threads can be enabled simply by setting the property:

```yaml
spring:
  threads:
    virtual:
      enabled: true
```

When enabled:
1. Embedded web servers (Tomcat, Jetty) use virtual threads to process incoming HTTP requests.
2. Spring MVC asynchronous request processing uses virtual threads.
3. `@Async` methods and `TaskExecutor` beans use `SimpleAsyncTaskExecutor` configured with virtual threads.
4. RabbitMQ and Kafka message listeners use virtual thread task executors.
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/actuator/observability.html",
        spring_version="3.4.0",
        module="actuator",
        title="Production-ready Features: Observability & Tracing",
        content="""# Production-ready Features: Observability & Tracing

Spring Boot actuator provides comprehensive observability integration with Micrometer and Micrometer Tracing.

## Distributed Tracing Configuration
When adding a tracing starter (such as `micrometer-tracing-bridge-brave` or `micrometer-tracing-bridge-otel`), Spring Boot configures tracing automatically.

```yaml
management:
  tracing:
    sampling:
      probability: 1.0
  zipkin:
    tracing:
      endpoint: "http://localhost:9411/api/v2/spans"
```

Traces and spans automatically correlate with logs using `logging.pattern.level` with traceId and spanId injection.
""",
    ),
    RawDocument(
        doc_url="https://docs.spring.io/spring-boot/reference/io/rest-client.html",
        spring_version="3.4.0",
        module="web",
        title="Calling REST Services: RestClient vs WebClient",
        content="""# Calling REST Services: RestClient vs WebClient

Spring Boot provides auto-configured builders for making HTTP requests to external REST services: RestClient, WebClient, and RestTemplate.

## RestClient vs WebClient Comparison
In Spring Boot 3.2+ and 3.4, `RestClient` is the recommended synchronous, blocking HTTP client. It offers a modern fluent API similar to `WebClient`, but without requiring reactive dependencies (Project Reactor):

1. **RestClient**: Synchronous and blocking. Designed for Spring MVC applications. Replaces the legacy `RestTemplate` with a fluent, declarative API.
2. **WebClient**: Asynchronous and non-blocking. Part of `spring-boot-starter-webflux`. Ideal for reactive microservices and streaming HTTP calls.
3. **RestTemplate**: Legacy synchronous client maintained in maintenance mode. New code should use `RestClient`.

## Using RestClient in Spring Boot 3.x
Inject the pre-configured `RestClient.Builder` bean to create customized client instances:

```java
package com.example.demo.client;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class GitHubClientService {

    private final RestClient restClient;

    public GitHubClientService(RestClient.Builder builder) {
        this.restClient = builder
            .baseUrl("https://api.github.com")
            .defaultHeader("Accept", "application/vnd.github.v3+json")
            .build();
    }

    public RepositoryResponse getRepository(String owner, String repo) {
        return this.restClient.get()
            .uri("/repos/{owner}/{repo}", owner, repo)
            .retrieve()
            .body(RepositoryResponse.class);
    }
}
```

## Using WebClient in Reactive Applications
For reactive applications, use `WebClient.Builder`:

```java
package com.example.demo.client;

import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
public class ReactiveGitHubService {

    private final WebClient webClient;

    public ReactiveGitHubService(WebClient.Builder builder) {
        this.webClient = builder.baseUrl("https://api.github.com").build();
    }

    public Mono<RepositoryResponse> getRepository(String owner, String repo) {
        return this.webClient.get()
            .uri("/repos/{owner}/{repo}", owner, repo)
            .retrieve()
            .bodyToMono(RepositoryResponse.class);
    }
}
```
""",
    ),
]
