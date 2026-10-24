"""Phase 1 Smoke Test Script: Validates 10 questions (8 factual + 2 abstention)."""

import asyncio
import sys
import time
from typing import Any

import httpx

TEST_CASES = [
    {
        "id": "Q1",
        "category": "factual",
        "topic": "Actuator Health",
        "query": "Como expor e habilitar detalhes do endpoint /actuator/health no Spring Boot 3.x?",
        "expected_keywords": ["actuator", "health", "management.endpoint.health.show-details", "always"],
    },
    {
        "id": "Q2",
        "category": "factual",
        "topic": "Spring MVC REST Controller",
        "query": "Como criar um REST Controller com Spring MVC retornando JSON no Spring Boot 3?",
        "expected_keywords": ["@RestController", "@GetMapping"],
    },
    {
        "id": "Q3",
        "category": "factual",
        "topic": "Connection Pool HikariCP",
        "query": "Qual é o connection pool padrão no Spring Boot 3 e como configurar o tamanho máximo do pool?",
        "expected_keywords": ["HikariCP", "maximum-pool-size"],
    },
    {
        "id": "Q4",
        "category": "factual",
        "topic": "Spring Security 6",
        "query": "Como configurar o SecurityFilterChain no Spring Boot 3 sem usar o WebSecurityConfigurerAdapter descontinuado?",
        "expected_keywords": ["SecurityFilterChain", "@Bean", "authorizeHttpRequests"],
    },
    {
        "id": "Q5",
        "category": "factual",
        "topic": "Auto-configuration Imports",
        "query": "Onde e como declarar classes de auto-configuração customizadas no Spring Boot 3.x?",
        "expected_keywords": [
            "META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports",
            "@AutoConfiguration",
        ],
    },
    {
        "id": "Q6",
        "category": "factual",
        "topic": "Configuration Properties",
        "query": "Como fazer bind de propriedades tipadas de configuração usando @ConfigurationProperties no Spring Boot 3?",
        "expected_keywords": ["@ConfigurationProperties", "prefix"],
    },
    {
        "id": "Q7",
        "category": "factual",
        "topic": "Virtual Threads",
        "query": "Como habilitar Virtual Threads (Project Loom) no Spring Boot 3.x e o que é afetado?",
        "expected_keywords": ["spring.threads.virtual.enabled", "virtual"],
    },
    {
        "id": "Q8",
        "category": "factual",
        "topic": "Distributed Tracing",
        "query": "Como configurar Distributed Tracing e amostragem de traces no Spring Boot Actuator?",
        "expected_keywords": ["management.tracing.sampling.probability", "tracing"],
    },
    {
        "id": "Q9",
        "category": "abstention",
        "topic": "Out-of-scope: Kubernetes Istio",
        "query": "Como configurar um cluster Kubernetes com Istio Service Mesh usando Helm charts e EnvoyFilter?",
        "expected_keywords": [],
    },
    {
        "id": "Q10",
        "category": "abstention",
        "topic": "Out-of-scope: React Redux",
        "query": "Qual é a sintaxe para criar um componente React com Redux Toolkit e TypeScript utilizando createSlice?",
        "expected_keywords": [],
    },
]


async def run_smoke_test(api_url: str = "http://localhost:8000") -> dict[str, Any]:
    """Execute smoke test against FastAPI server or in-memory service."""
    print("=" * 80)
    print(" SPRINGDOCS ASSISTANT - PHASE 1 SMOKE TEST SUITE")
    print(f" Target Endpoint: {api_url}/ask")
    print("=" * 80)

    # First test /health
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            health_res = await client.get(f"{api_url}/health")
            print(f"[*] Health Check HTTP Status: {health_res.status_code}")
            print(f"[*] Health Details: {health_res.text}")
        except Exception as exc:
            print(f"[-] Could not connect to {api_url}/health: {exc}")
            print("[-] Ensure the FastAPI application is running (e.g. `uv run uvicorn src.main:app --port 8000`).")
            sys.exit(1)

    results = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []

    factual_citations_count = 0
    abstention_success_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        for item in TEST_CASES:
            qid = item["id"]
            cat = item["category"]
            topic = item["topic"]
            query = item["query"]

            print(f"\n[{qid}] [{cat.upper()}] Topic: {topic}")
            print(f"     Query: {query}")

            start_t = time.perf_counter()
            response = await client.post(
                f"{api_url}/ask",
                json={"query": query},
            )
            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

            if response.status_code != 200:
                print(f"[-] Request failed with status {response.status_code}: {response.text}")
                continue

            data = response.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            ret_ms = data.get("retrieval_latency_ms", 0.0)
            gen_ms = data.get("generation_latency_ms", 0.0)
            tot_ms = data.get("total_latency_ms", elapsed_ms)
            is_abstained = data.get("is_abstained", False)

            retrieval_latencies.append(ret_ms)
            generation_latencies.append(gen_ms)
            total_latencies.append(tot_ms)

            # Evaluate success criteria
            if cat == "factual":
                has_citation = len(citations) > 0 or "[Fonte:" in answer
                if has_citation:
                    factual_citations_count += 1
                status_str = "PASS (Cited)" if has_citation else "WARN (No citation)"
            else:  # abstention
                if is_abstained or "não encontrei" in answer.lower():
                    abstention_success_count += 1
                    status_str = "PASS (Abstained)"
                else:
                    status_str = "FAIL (Hallucinated)"

            print(f"     Status: {status_str}", flush=True)
            print(f"     Latency: {tot_ms}ms (Retrieval: {ret_ms}ms, Generation: {gen_ms}ms)", flush=True)
            if citations:
                print(f"     Citations: {citations}", flush=True)
            snippet = answer.replace("\n", " ")[:160]
            print(f"     Answer Snippet: {snippet}...", flush=True)

            results.append({
                "id": qid,
                "category": cat,
                "topic": topic,
                "status": status_str,
                "retrieval_latency_ms": ret_ms,
                "generation_latency_ms": gen_ms,
                "total_latency_ms": tot_ms,
                "citations": citations,
                "is_abstained": is_abstained,
                "answer": answer,
            })

    total_factual = sum(1 for t in TEST_CASES if t["category"] == "factual")
    total_abstention = sum(1 for t in TEST_CASES if t["category"] == "abstention")

    citation_rate = round((factual_citations_count / total_factual) * 100, 1)
    abstention_rate = round((abstention_success_count / total_abstention) * 100, 1)
    avg_retrieval = round(sum(retrieval_latencies) / len(retrieval_latencies), 2) if retrieval_latencies else 0.0
    avg_generation = round(sum(generation_latencies) / len(generation_latencies), 2) if generation_latencies else 0.0
    avg_total = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0.0

    print("\n" + "=" * 80)
    print(" PHASE 1 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated : {len(TEST_CASES)}")
    print(f"Correct Citation Rate     : {citation_rate}% ({factual_citations_count}/{total_factual})")
    print(f"Safe Abstention Rate      : {abstention_rate}% ({abstention_success_count}/{total_abstention})")
    print(f"Average Retrieval Latency : {avg_retrieval} ms")
    print(f"Average Generation Latency: {avg_generation} ms")
    print(f"Average Total Latency     : {avg_total} ms")
    print("=" * 80)

    summary = {
        "total_questions": len(TEST_CASES),
        "citation_rate": citation_rate,
        "abstention_rate": abstention_rate,
        "avg_retrieval_ms": avg_retrieval,
        "avg_generation_ms": avg_generation,
        "avg_total_ms": avg_total,
        "results": results,
    }

    return summary


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
