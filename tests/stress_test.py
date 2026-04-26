#!/usr/bin/env python3
# stress_test.py is NOT a pytest test.
# Run manually: python tests/stress_test.py
# This file exists in tests/ for organisation only and is excluded from any CI pipeline.
"""
Stress Test per selenium-llm-engine
===================================
Invia prompt lunghi (3-4 chunk) con ramp-up per testare:
- Tempistiche medie per engine
- Solidità del sistema sotto pressione
- Coda FIFO tra engine
"""

import asyncio
import json
import time
import sys
import statistics
from dataclasses import dataclass, field
from typing import Any

import httpx


BASE_URL = "http://localhost:14848"


LONG_PROMPTS_GEMINI = [
    """Analizza il concetto di architettura software a microservizi, concentrandoti su:\n\n1. Definizione e caratteristiche principali: spiega cosa sono i microservizi, come differiscono dalle architetture monolitiche tradizionali, e quali sono i principi fondamentali che li guidano. Includi discussioni su bounded context, single responsibility e come queste idee influenzano la progettazione dei servizi.\n\n2. Vantaggi e svantaggi nell'implementazione reale: elenca i principali benefici come scalabilità indipendente, deployment autonomo, isolamento dei guasti e tecnologia eterogenea. Discoti anche gli svantaggi significativi come la complessità operativa, la gestione delle transazioni distribuite, il debugging tra servizi, e il costo dell'infrastruttura di rete.\n\n3. Pattern di comunicazione inter-servizio: descrivi i diversi approcci disponibili - sincrono (REST, gRPC) versus asincrono (message queue, event-driven), e quando è appropriato utilizzare ciascun pattern. Includi considerazioni su CQRS, saga pattern e event sourcing.\n\n4. Monitoring e observability in ambiente distribuito: spiega come implementare logging distribuito, tracing centralizzato e metriche aggregate quando si hanno decine o centinaia di servizi interconnessi.""",

    """Fornisci un'analisi approfondita delle strategie di machine learning per la classificazione di testo in ambito NLP, con focus su:\n\n1. Approcci tradizionali vs deep learning: confronta Naive Bayes, SVM e Random Forest con le reti neurali LSTM, GRU, Transformer e BERT-like models. Descrivi quando è appropriato usare ciascun approccio basandosi su dimensione del dataset, risorse computazionali disponibili e requisiti di interpretability.\n\n2. Feature engineering per NLP: spiega rappresentazioni come TF-IDF, bag-of-words, word embeddings (Word2Vec, GloVe, FastText) e contextual embeddings. Discoti l'importanza della pre-elaborazione del testo: tokenization, stemming, lemmatization, stop words removal e named entity recognition.\n\n3. Transfer learning e fine-tuning: illustra come i grandi modelli pre-addestrati abbiano rivoluzionato il campo NLP. Descrivi pipeline per pre-addestramento su corpora massivi, domain adaptation e task-specific fine-tuning con esempi concreti.\n\n4. Valutazione e hyperparameter tuning: elenca metriche come accuracy, precision, recall, F1-score, AUC-ROC e quando usare ciascuna. Discoti tecniche di cross-validation, grid search, random search e Bayesian optimization per ottimizzare gli iperparametri.""",

    """Descrivi nel dettaglio il ciclo di vita dello sviluppo software Agile, con particolare attenzione a:\n\n1. Principi fondamentali del Manifesto Agile: analizza i 4 valori e i 12 principi del Manifesto Agile. Spiega come questi valori influenzino le pratiche quotidiane dello sviluppo software, incluso il concetto di "individui e interazioni sopra processi e strumenti" e "rispondere al cambiamento sopra seguire un piano".\n\n2. Framework e metodologie agili: confronta Scrum, Kanban, XP (Extreme Programming) e SAFe. Descrivi ruoli (Product Owner, Scrum Master, Development Team), eventi (Sprint Planning, Daily Standup, Sprint Review, Retrospective) e artifact (Product Backlog, Sprint Backlog, Increment) tipici di Scrum.\n\n3. User Stories e criteria di accettazione: spiega come scrivere user stories effective seguendo il formato "As a [role], I want [goal] so that [benefit]". Descrivi INVEST criteria (Independent, Negotiable, Valuable, Estimable, Small, Testable) e come definire acceptance criteria non ambigui.\n\n4. Continuous Integration/Continuous Deployment: illustra l'importanza di CI/CD nel contesto Agile, includendo automated testing, build automation, feature branching strategies, e deployment pipelines. Discoti come ridurre il cycle time e aumentare la frequency of releases mantenendo qualità.""",

    """Esplora le best practices per la sicurezza informatica in ambito cloud computing, concentrandoti su:\n\n1. Shared responsibility model: spiega come la responsabilità della sicurezza sia divisa tra cloud provider e cliente nei modelli IaaS, PaaS e SaaS. Descrivi cosa il provider garantisce tipicamente (physical security, infrastructure security) versus cosa è responsabilità del cliente (data classification, identity management, application security).\n\n2. Identity and Access Management (IAM): analizza i principi di least privilege e need-to-know, ruoli e policies, multi-factor authentication, e Just-In-Time access. Discoti service accounts, service principals e come gestire le credenziali in ambienti cloud-native.\n\n3. Data encryption e secure storage: descrivi encryption at rest e in transit, key management services (Azure Key Vault, AWS KMS, GCP Cloud KMS), envelope encryption e field-level encryption. Discoti anche data residency e compliance requirements come GDPR.\n\n4. Network security e microsegmentation: spiega come progettare reti cloud sicure con subnets pubbliche/private, Network Security Groups, Virtual Private Networks, e microsegmentation per limitare la blast radius in caso di breach. Include discussioni su Zero Trust networking.""",

    """Fornisci una guida completa all'implementazione di sistemi di database distribuiti, con focus su:\n\n1. Teorema CAP e conseguenze pratiche: spiega il teorema CAP (Consistency, Availability, Partition Tolerance) e come nessun sistema distribuito può garantire tutte e tre le proprietà simultaneamente. Discoti le implicazioni per la scelta tra sistemi CP e AP, con esempi come Cassandra (AP), MongoDB (CP/CA configurabile), e Spanner (CP).\n\n2. Replication strategies: analizza synchronous vs asynchronous replication, leader-based vs leaderless replication, e multi-leader setups. Descrive come gestire fail-over automatico, split-brain scenarios, e consensus algorithms come Raft e Paxos.\n\n3. Sharding e data partitioning: illustra diverse strategie di sharding - hash-based, range-based, e directory-based. Discoti i challenges associati: resharding, hot spots, e cross-shard transactions. Include considerazioni su consistent hashing.\n\n4. ACID vs BASE tradeoffs: confronta il modello ACID tradizionale dei database relazionali con il modello BASE (Basically Available, Soft state, Eventually consistent) dei sistemi NoSQL. Spiega quando ciascun modello è appropriato e come molti sistemi moderni offrono tunable consistency.""",

    """Analizza le tecniche di ottimizzazione delle performance per applicazioni web ad alta intensità di traffico:\n\n1. Frontend optimization: spiega lazy loading, code splitting, tree shaking, e caching strategies (CDN, browser cache, service workers). Discoti image optimization, critical CSS extraction, e render-blocking resources. Include discussioni su Core Web Vitals (LCP, FID, CLS) e come ottimizzarli.\n\n2. Backend performance: analizza database query optimization, connection pooling, caching strategies (Redis, Memcached), e asynchronous processing. Descrive profiling tools, identifying bottlenecks, e techniques come query denormalization, indexing strategies, e read replicas.\n\n3. Scalability patterns: illustra horizontal vs vertical scaling, load balancing strategies (round-robin, least connections, IP hash), e auto-scaling. Discoti stateless application design, session management, e geographic distribution per global audiences.\n\n4. API design per performance: spiega GraphQL vs REST, batched requests, field projection, e pagination strategies. Discoti rate limiting, API versioning, e caching headers (Cache-Control, ETag, Last-Modified). Include considerationi su WebSocket per real-time communication.""",

    """Esplora il domain-driven design (DDD) come approccio alla modellazione software complessa:\n\n1. Bounded Contexts e Ubiquitous Language: spiega come identificare bounded contexts all'interno di un dominio, come definire una ubiquitous language condivisa tra team, e come gestire context maps. Discoti le relationships tra contexts: shared kernel, customer-supplier, conformist, e antipatterns come big ball of mud.\n\n2. Aggregates, Entities, e Value Objects: analizza come modellare dominio usando questi pattern. Spiega aggregate roots, invariants enforcement, e transaction boundaries. Discoti come entities differiscono da value objects in terms of identity e immutability.\n\n3. Domain Events e Event Sourcing: illustra come usare domain events per catturare state changes, CQRS pattern per separare read/write models, e event sourcing per persistere non lo stato ma la sequence di events. Discoti benefits come completa audit trail e temporal queries.\n\n4. Strategic vs Tactical Design: confronta i patterns strategici (bounded contexts, context mapping) con quelli tattici (entities, value objects, services, repositories, factories). Discoti quando usare ciascuno e come mantenere focus sul business value piuttosto che technical complexity.""",

    """Descrivi l'architettura e le considerazioni di sicurezza per applicazioni containerizzate con Kubernetes:\n\n1. Container security fundamentals: analizza secure container images (minimal base images, multi-stage builds), container runtime security, e vulnerability scanning. Discoti principle of least privilege per container permissions, seccomp, AppArmor, e SELinux profiles.\n\n2. Kubernetes network policies: spiega default-deny network policies, microsegmentation tra pods, e DNS security. Discoti service mesh considerations (Istio, Linkerd) per mTLS, traffic encryption, e fine-grained access control tra servizi.\n\n3. Secrets management: analizza Kubernetes secrets vs external secrets (HashiCorp Vault, AWS Secrets Manager), encryption at rest per secrets, e RBAC per secrets access. Discoti rotated credentials, secret propagation, e best practices per injection nei pods.\n\n4. RBAC e authentication/authorization: illustra roles, cluster roles, role bindings, e service accounts. Discoti external identity providers (OIDC, LDAP integration), admission controllers, e pod security policies/standards.""",

    """Fornisci un'analisi dettagliata dei pattern di messaging asincrono per sistemi distribuiti:\n\n1. Message brokers e code: confronta RabbitMQ, Apache Kafka, Amazon SQS/SNS, e Azure Service Bus. Spiega publish/subscribe vs point-to-point, message durability, e dead letter queues. Discoti when to use each basandosi su throughput requirements, ordering guarantees, e message retention needs.\n\n2. Outbox pattern per reliability: illustra il problema dei dual writes in sistemi distribuiti, e come il outbox pattern risolve questo problema usando un database transaction per scrivere sia il business data che il message outbox. Discoti implementation strategies e variants come the transactional outbox pattern.\n\n3. Idempotency e deduplication: spiega perché i consumers devono essere idempotenti quando si processano messaggi, e le strategies per deduplication (message IDs, database unique constraints, content-based hashing). Discoti exactly-once semantics e i suoi tradeoffs.\n\n4. Circuit breaker e retry patterns: analizza come proteggere i system components da cascading failures usando circuit breakers. Descrive retry strategies con exponential backoff, jitter, e dead letter queue per failed messages. Include discussioni su bulkheads e fallback mechanisms.""",

    """Esplora le metodologie e gli strumenti per il testing di software complesso:\n\n1. Test pyramid e testing strategy: spiega la test pyramid (unit, integration, e2e tests) e perché è importante. Discoti tradeoffs tra different levels of testing, the cost of different test types, e come bilanciare coverage vs speed. Include discussioni su mutation testing e code coverage metrics.\n\n2. Property-based testing: illustra il concetto di testing against specifications piuttosto che specific examples. Descrive generators, shrinking, e property definitions. Discoti quando property-based testing è particolarmente valuable (stateful systems, complex data transformations, protocol conformance).\n\n3. Contract testing e consumer-driven contracts: analizza il problema degli integration tests fragile, e come contract testing lo risolve. Spiega provider contracts vs consumer contracts, Pact framework, e CDC testing. Discoti benefits come independent service evolution e reduced integration testing burden.\n\n4. Chaos engineering: illustra come introdurre deliberatamente failures per testare la resilience. Descrive Netflix Chaos Monkey, steady-state hypothesis, e blast radius. Discoti what to test (network failures, resource exhaustion, dependency failures) e how to measure impact."""
]

LONG_PROMPTS_CHATGPT = [
    """Spiega in modo approfondito il concetto di containerizzazione e virtualizzazione, analizzando:\n\n1. Differenze tra VM e container: confronta l'approccio tradizionale delle macchine virtuali con quello dei container. Descrivi come le VM emulano hardware fisico tramite hypervisor (Type 1 e Type 2) e come questo comporta overhead significativo, versus i container che condividono il kernel del sistema operativo host ma mantengono isolamento a livello di processo.\n\n2. Tecnologie di containerizzazione: analizza Docker come standard de facto, le differenze tra Docker e containerd, e il ruolo del OCI (Open Container Initiative). Spiega BuildKit, multi-stage builds, e le differenze tra immagini container e orchestrazione con Kubernetes, Docker Swarm, e podman.\n\n3. Namespaces e cgroups: illustra i Linux kernel features che rendono possibile la containerizzazione - PID namespace, network namespace, mount namespace, user namespace, e cgroups per resource limiting. Spiega come questi meccanismi creano l'illusione di isolamento completo.\n\n4. Best practices per immagini container production-ready: descrivi minimizing image size, using specific tags, multi-stage builds per compilazione分离, non-root users per sicurezza, health checks, e proper logging. Discoti layer caching e build optimization strategies.""",

    """Analizza le strategie di data management nell'era del Big Data:\n\n1. Data lakes vs data warehouses: confronta questi due approcci alla gestione dati enterprise. Spiega come i data warehouses tradizionali (schema-on-write) differiscono dai data lakes (schema-on-read), e quando ciascuno è appropriato. Discoti modern approaches come lakehouse architectures che combinano i benefici di entrambi.\n\n2. ETL vs ELT pipelines: analizza le differenze tra Extract-Transform-Load e Extract-Load-Transform, e come il cloud data warehousing ha spostato il paradigma verso ELT. Descrive tools come dbt, Airbyte, Fivetran, e come modern data stacks sono costruiti.\n\n3. Streaming data processing: illustra l'architettura lambda e kappa per processing real-time. Confronta Apache Kafka con Amazon Kinesis, Apache Flink con Apache Spark Streaming, e spiega windowing, state management, e exactly-once semantics.\n\n4. Data governance e compliance: spiega data lineage, cataloging, e discovery. Analizza regulatory requirements come GDPR, CCPA, e HIPAA e il loro impatto sulla data architecture. Discoti anonymization techniques, data masking, e privacy-preserving analytics.""",

    """Fornisci una panoramica completa delle API REST e del loro design:\n\n1. Principi REST: analizza i 6 vincoli architetturali REST (client-server, stateless, cacheable, layered system, code on demand, uniform interface). Spiega come questi principi guidano il design di web services scalabili e come deviare da REST porta a RPC o GraphQL.\n\n2. HTTP methods e status codes: illustra l'uso appropriato di GET, POST, PUT, PATCH, DELETE e come rispettare idempotency e safety properties. Descrive common status codes (2xx, 4xx, 5xx) e cuándo usare ciascuno per una comunicazione chiara tra client e server.\n\n3. API versioning e evolution: analizza diverse strategie di versioning (URL path, query parameter, header, content negotiation) e i loro tradeoffs. Spiega come gestire breaking changes mantenendo backward compatibility, e come i consumer-driven contracts aiutano in questo processo.\n\n4. Security best practices: descrivi OAuth 2.0 e OpenID Connect per authentication, JWT per stateless tokens, rate limiting, input validation, e CORS configuration. Discoti API security headers, request signing, e bot protection mechanisms.""",

    """Esplora l'implementazione di sistemi di autenticazione moderni:\n\n1. OAuth 2.0 in profondità: analizza grant types (authorization code, implicit, client credentials, refresh token) e i loro use cases appropriati. Spiega PKCE per public clients, scope-based access control, e token lifetime management. Discoti le differenze tra access tokens e refresh tokens.\n\n2. JWT e token management: illustra la struttura JWT (header, payload, signature), claims standard, e HS256 vs RS256 signing algorithms. Discoti token storage (memory, localStorage, httpOnly cookies), refresh token rotation, e token revocation strategies.\n\n3. Multi-factor authentication: analizza i tre factor types (something you know, have, are) e le combinazioni per strong authentication. Spiega TOTP, push notifications, hardware keys (FIDO2/WebAuthn), e SMS-based MFA con le loro security/e UX tradeoffs.\n\n4. SSO e identity federation: descrivi SAML 2.0 e OpenID Connect come protocolli di federazione. Spiega identity providers, service providers, e come implementare SSO aziendale con directory integration (LDAP, Active Directory). Include discussioni on identity lifecycle management.""",

    """Analizza le metodologie DevOps e la cultura associated:\n\n1. Three ways of DevOps: illustra i principi foundational del DevOps - Flow, Feedback, e Continuous Learning. Spiega come questi principi informano tutte le pratiche DevOps e creano un sistema di miglioramento continuo throughput delivery pipeline.\n\n2. CALMS framework: analizza Culture, Automation, Lean, Measurement, e Sharing come pilastri del DevOps. Spiega come questi elementi lavorano insieme e come coltivare una cultura di collaborazione tra development e operations teams.\n\n3. SRE vs DevOps: confronta Site Reliability Engineering con DevOps tradizionale. Spiega i principi SRE (SLIs/SLOs/SLAs, error budgets, Toil) e come differiscono dalla mentalità tradizionale operations. Discoti quando scegliere ogni approccio.\n\n4. DevOps toolchain: illustra il continuum di tool dagli IDE ai production monitoring. Descrive version control, CI/CD, artifact management, container orchestration, e observability stacks. Discoti come selezionare tool che supportano la cultura piuttosto che sostituirla.""",

    """Fornisci una guida ai sistemi di messaggistica e code di messaggi:\n\n1. Message queue patterns: analizza point-to-point (queues) vs publish-subscribe (topics). Spiega message ordering, delivery guarantees (at-most-once, at-least-once, exactly-once), e dead letter queues per failed message handling.\n\n2. Enterprise integration patterns: illustra channel adapters, message routers, message transformers, e endpoint. Spiega come questi pattern permettono di costruire integration flows complessi con componenti semplici, e quali tools li implementano (Spring Integration, Mule ESB, Camel).\n\n3. Saga pattern per transazioni distribuite: analizza come gestire business transactions spanning multiple services usando sagas instead of distributed transactions. Descrive choreography vs orchestration approaches, compensatng transactions, e come prevenire cascade failures.\n\n4. Event-driven architectures: spiega il transition da request-driven a event-driven design. Analizza event sourcing (persist events, not state), CQRS (separate read/write models), e come questi pattern supportano eventual consistency e high scalability.""",

    """Esplora le pratiche di secure coding e software security:\n\n1. OWASP Top 10: analizza i 10 rischi di security più critici per applicazioni web secondo OWASP. Spiega injection (SQL, XSS, command injection), broken authentication, sensitive data exposure, e le mitigation strategies per ciascuno.\n\n2. Input validation e sanitization: illustra perché input validation è la prima linea di difesa, whitelist vs blacklist approaches, e parameterized queries per prevenire injection. Discoti output encoding e context-specific escaping per XSS prevention.\n\n3. Authentication e session management: analizza secure password storage (hashing algorithms, salt, cost factors), session fixation e hijacking, e secure session token generation. Spiega account recovery procedures e password requirements tradeoffs.\n\n4. Cryptography fundamentals: descrivi symmetric vs asymmetric encryption, hashing algorithms (SHA-256, bcrypt, Argon2), e MAC/HMAC. Spiega key management, certificate authorities, e TLS handshake flow. Discoti common crypto failures e best practices.""",

    """Analizza l'observability nei sistemi distribuiti moderni:\n\n1. Three pillars of observability: illustra logs, metrics, e traces come complementari sources di telemetry. Spiega come ciascuno fornisce visibility diversa e perché tutti e tre sono necessari per debuggare sistemi complessi.\n\n2. Structured logging: analizza il shift da log text unstructured a structured logs (JSON). Spiega log levels, correlation IDs per request tracing, e log aggregation systems (ELK, Loki, CloudWatch). Discoti sampling strategies per high-volume systems.\n\n3. Distributed tracing: illustra come funziona il tracing distributed (trace IDs propagati tra services), e tools come Jaeger, Zipkin, e cloud-native solutions. Spiega tail-based sampling per capture rare events, e come correlation con logs completa il picture.\n\n4. SLOs e alert design: analizza come definire Service Level Indicators e Objectives, error budgets, e burn rate alerts. Spiega why-to-alert e come evitare alert fatigue. Discoti runbooks e incident response procedures per actionable alerts.""",

    """Descrivi le metodologie per il design di database relazionali ad alte prestazioni:\n\n1. Normalization vs denormalization: illustra le normal forms (1NF through 3NF, BCNF) e quando normalization è appropriata versus quando denormalization migliora performance. Spiega materializzed views come compromise pragmatico.\n\n2. Indexing strategies: analizza B-tree vs hash indexes, covering indexes, partial indexes, e expression indexes. Spiega index selectivity e statistics management. Discoti clustered vs non-clustered indexes e come they impact query plans.\n\n3. Query optimization: illustra query plan analysis con EXPLAIN, join order optimization, e subquery flattening. Spiega statistics-based optimization e come different execution plans impact performance. Discoti common anti-patterns che causano performance issues.\n\n4. Concurrency control: analizza locking vs MVCC, lock granularity, e deadlocks. Spiega optimistic vs pessimistic concurrency models e quando ciascuno è appropriato. Discoti isolation levels e il loro impatto su performance e correctness.""",

    """Esplora i pattern di design per applicazioni cloud-native:\n\n1. Twelve-factor app methodology: analizza i 12 fattori per application development in cloud environments - codebase, dependencies, config, backing services, build/release/run, processes, port binding, concurrency, disposability, dev/prod parity, logs, admin processes. Spiega come each factor contributes to scalability e deployability.\n\n2. Strangler fig pattern: illustra come modernizzare applicazioni legacy gradualmente. Spiega come costruire nuove features intorno al sistema esistente, estrarre capabilities pezzo per pezzo, e ridurre il rischio di migration failures.\n\n3. Anti-corruption layer: analizza come prevenire legacy system design da contaminare new architecture quando integrating with existing systems. Spiega adapter patterns, anti-corruption layer implementation, e bounded context boundaries.\n\n4. Sidecar e ambassador patterns: illustra come estendere container functionality senza modificare application code. Spiega sidecar containers per logging, proxying, e configuration management, e ambassador per service discovery e circuit breaking."""
]

MISC_ENGINES = ["claude", "copilot", "grok", "perplexity", "stepfun"]

SHORT_PROMPTS_MISC = [
    """Spiega brevemente il concetto di inversion of control e dependency injection nei pattern di design software moderno. Includi esempi pratici di implementazione in Python e quando questo approccio risulta vantaggioso rispetto a un design tradizionale.""",

    """Descrivi i principali pattern di creazione di oggetti: factory method, abstract factory, builder e prototype. Per ciascuno, indica il caso d'uso tipico, un esempio di implementazione in pseudocodice e i pro/contro rispetto agli altri pattern.""",

    """Analizza il concetto di eventual consistency nei database distribuiti moderni. Spiega come Cassandra, DynamoDB e Cosmos DB gestiscono la consistenza configurabile, quali garanzie offrono e quali tradeoffs comportano in termini di disponibilità e prestazioni.""",

    """Fornisci un confronto tra GraphQL e REST come approcci al design di API, analizzando performance, caching, type safety, schema evolution e ecosystem per applicazioni enterprise di medie-grandi dimensioni.""",

    """Esplora il concetto di circuit breaker pattern per la resilienza di sistemi distribuiti. Spiega gli stati (closed, open, half-open), la soglia di trigger, le strategie di fallback e come implementare un circuit breaker robust in un'applicazione Python."""
]


@dataclass
class PromptResult:
    engine: str
    prompt_index: int
    queue_wait_ms: float
    total_ms: float
    success: bool
    error: str = ""
    model: str = ""


@dataclass
class EngineStats:
    name: str
    results: list[PromptResult] = field(default_factory=list)

    @property
    def successful(self) -> list[PromptResult]:
        return [r for r in self.results if r.success]

    @property
    def avg_queue_wait_ms(self) -> float:
        suc = self.successful
        return statistics.mean(r.queue_wait_ms for r in suc) if suc else 0

    @property
    def avg_total_ms(self) -> float:
        suc = self.successful
        return statistics.mean(r.total_ms for r in suc) if suc else 0

    @property
    def min_total_ms(self) -> float:
        suc = self.successful
        return min(r.total_ms for r in suc) if suc else 0

    @property
    def max_total_ms(self) -> float:
        suc = self.successful
        return max(r.total_ms for r in suc) if suc else 0

    @property
    def median_total_ms(self) -> float:
        suc = self.successful
        return statistics.median(r.total_ms for r in suc) if suc else 0

    @property
    def p95_total_ms(self) -> float:
        suc = self.successful
        if len(suc) < 2:
            return 0
        sorted_ms = sorted(r.total_ms for r in suc)
        idx = int(len(sorted_ms) * 0.95)
        return sorted_ms[min(idx, len(sorted_ms) - 1)]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results) * 100

    @property
    def stddev_total_ms(self) -> float:
        suc = self.successful
        if len(suc) < 2:
            return 0.0
        return statistics.stdev(r.total_ms for r in suc)


async def send_prompt(
    client: httpx.AsyncClient,
    engine: str,
    prompt: str,
    prompt_index: int,
    request_start: float,
) -> PromptResult:
    result = PromptResult(engine=engine, prompt_index=prompt_index, queue_wait_ms=0, total_ms=0, success=False)

    try:
        resp = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": engine, "messages": [{"role": "user", "content": prompt}]},
            timeout=180.0,
        )
        result.total_ms = (time.time() - request_start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            result.success = True
            result.model = data.get("model", "unknown")
            result.queue_wait_ms = data.get("elapsed_ms", result.total_ms)
            # estimate queue wait as total - estimate response time
            if result.queue_wait_ms > result.total_ms:
                result.queue_wait_ms = 0
        else:
            result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except httpx.TimeoutException:
        result.total_ms = (time.time() - request_start) * 1000
        result.error = "TIMEOUT"
    except Exception as exc:
        result.total_ms = (time.time() - request_start) * 1000
        result.error = str(exc)[:200]

    return result


async def run_stress_test():
    print("=" * 80)
    print("STRESS TEST - selenium-llm-engine")
    print("=" * 80)
    print()

    all_stats: dict[str, EngineStats] = {}

    async with httpx.AsyncClient() as client:
        ping_resp = await client.get(f"{BASE_URL}/api/ping")
        print(f"Ping: {ping_resp.json()}")
        print()

        engines_resp = await client.get(f"{BASE_URL}/api/engines")
        engines_data = engines_resp.json()
        available = [e["name"] for e in engines_data["data"]]
        print(f"Available engines: {available}")
        print()

        for engine in MISC_ENGINES:
            if engine in available:
                all_stats[engine] = EngineStats(name=engine)

        for p in LONG_PROMPTS_GEMINI:
            all_stats.setdefault("gemini", EngineStats(name="gemini"))
        for p in LONG_PROMPTS_CHATGPT:
            all_stats.setdefault("chatgpt", EngineStats(name="chatgpt"))

        delays = [30, 20, 15, 10, 7, 5, 3, 2, 1, 0]
        gemini_tasks: list[asyncio.Task] = []
        chatgpt_tasks: list[asyncio.Task] = []
        misc_tasks: list[asyncio.Task] = []
        request_start_times: dict[int, float] = {}

        async def wrap_prompt(
            engine: str,
            prompt: str,
            prompt_index: int,
            future: asyncio.Future,
            task_id: int,
        ):
            request_start = time.time()
            request_start_times[task_id] = request_start
            result = await send_prompt(
                httpx.AsyncClient(),
                engine,
                prompt,
                prompt_index,
                request_start,
            )
            if not future.done():
                future.set_result(result)

        # ==========================
        # PHASE 1: GEMINI (10 prompts)
        # ==========================
        print("-" * 80)
        print("PHASE 1: Gemini (10 long prompts, ramp-up delays)")
        print("-" * 80)

        gemini_loop = asyncio.get_event_loop()
        gemini_futures: list[asyncio.Future] = []

        for i, prompt in enumerate(LONG_PROMPTS_GEMINI):
            future = asyncio.Future()
            gemini_futures.append(future)
            task_id = i
            t = asyncio.create_task(wrap_prompt("gemini", prompt, i, future, task_id))
            gemini_tasks.append(t)

            delay = delays[min(i, len(delays) - 1)]
            if delay > 0:
                print(f"  [Gemini #{i+1}] Scheduling with delay: {delay}s")
                await asyncio.sleep(delay)

        print(f"  All {len(gemini_tasks)} Gemini tasks dispatched, waiting for completions...")
        gemini_results = await asyncio.gather(*[f for f in gemini_futures])

        for r in gemini_results:
            if isinstance(r, PromptResult):
                all_stats["gemini"].results.append(r)

        print(f"  Gemini phase complete: {len(gemini_results)} completed")
        await asyncio.sleep(3)

        # ==========================
        # PHASE 2: CHATGPT (10 prompts)
        # ==========================
        print()
        print("-" * 80)
        print("PHASE 2: ChatGPT (10 long prompts, ramp-up delays)")
        print("-" * 80)

        chatgpt_futures: list[asyncio.Future] = []

        for i, prompt in enumerate(LONG_PROMPTS_CHATGPT):
            future = asyncio.Future()
            chatgpt_futures.append(future)
            task_id = 100 + i
            t = asyncio.create_task(wrap_prompt("chatgpt", prompt, i, future, task_id))
            chatgpt_tasks.append(t)

            delay = delays[min(i, len(delays) - 1)]
            if delay > 0:
                print(f"  [ChatGPT #{i+1}] Scheduling with delay: {delay}s")
                await asyncio.sleep(delay)

        print(f"  All {len(chatgpt_tasks)} ChatGPT tasks dispatched, waiting for completions...")
        chatgpt_results = await asyncio.gather(*[f for f in chatgpt_futures])

        for r in chatgpt_results:
            if isinstance(r, PromptResult):
                all_stats["chatgpt"].results.append(r)

        print(f"  ChatGPT phase complete: {len(chatgpt_results)} completed")
        await asyncio.sleep(3)

        # ==========================
        # PHASE 3: MISC ENGINES (5 prompts each)
        # ==========================
        print()
        print("-" * 80)
        print("PHASE 3: Misc engines (5 random prompts each, burst)")
        print("-" * 80)

        misc_futures: list[tuple[str, asyncio.Future]] = []

        for engine in MISC_ENGINES:
            if engine not in all_stats:
                continue
            for i, prompt in enumerate(SHORT_PROMPTS_MISC[:5]):
                future = asyncio.Future()
                misc_futures.append((engine, future))
                task_id = 200 + len(misc_futures)
                t = asyncio.create_task(wrap_prompt(engine, prompt, i, future, task_id))
                misc_tasks.append(t)
            delay = delays[min(len(misc_futures) // len([e for e in MISC_ENGINES if e in all_stats]), len(delays) - 1)]
            if delay > 0:
                print(f"  [{engine}] Scheduling {len([e for e in misc_futures if e[0] == engine])} tasks with delay: {delay}s")
                await asyncio.sleep(delay)

        print(f"  All {len(misc_tasks)} misc tasks dispatched, waiting for completions...")
        misc_results = await asyncio.gather(*[f for _, f in misc_futures])

        for (engine, _), r in zip(misc_futures, misc_results):
            if isinstance(r, PromptResult):
                all_stats[engine].results.append(r)

        print(f"  Misc phase complete: {len(misc_results)} completed")

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    for name, stats in sorted(all_stats.items(), key=lambda x: x[0]):
        print(f"{'='*60}")
        print(f"ENGINE: {name.upper()}")
        print(f"{'='*60}")
        print(f"  Total requests:   {len(stats.results)}")
        print(f"  Successful:       {len(stats.successful)} ({stats.success_rate:.1f}%)")
        print(f"  Failed:           {len(stats.results) - len(stats.successful)}")

        if stats.successful:
            print(f"  --- Timing (successful requests only) ---")
            print(f"  Avg response:     {stats.avg_total_ms:.0f}ms ({stats.avg_total_ms/1000:.1f}s)")
            print(f"  Median response:  {stats.median_total_ms:.0f}ms ({stats.median_total_ms/1000:.1f}s)")
            print(f"  Min response:     {stats.min_total_ms:.0f}ms ({stats.min_total_ms/1000:.1f}s)")
            print(f"  Max response:     {stats.max_total_ms:.0f}ms ({stats.max_total_ms/1000:.1f}s)")
            print(f"  P95 response:     {stats.p95_total_ms:.0f}ms ({stats.p95_total_ms/1000:.1f}s)")
            print(f"  StdDev:           {stats.stddev_total_ms:.0f}ms")
            if stats.avg_queue_wait_ms > 0:
                print(f"  Avg queue wait:   {stats.avg_queue_wait_ms:.0f}ms")

        failures = [r for r in stats.results if not r.success]
        if failures:
            print(f"  --- Failures ---")
            for f in failures:
                print(f"    Prompt #{f.prompt_index+1}: {f.error[:100]}")

        print()

    print()
    print("=" * 80)
    print("PER-REQUEST DETAIL")
    print("=" * 80)
    for name, stats in sorted(all_stats.items(), key=lambda x: x[0]):
        print(f"\n{name.upper()}:")
        for r in stats.results:
            status = "OK" if r.success else "FAIL"
            print(f"  #{r.prompt_index+1}: {status} | total={r.total_ms:.0f}ms | model={r.model or '-'} | err={r.error[:50] if r.error else '-'}")

    print()
    print("=" * 80)
    print("SYSTEM SOLIDITY REPORT")
    print("=" * 80)

    total_requests = sum(len(s.results) for s in all_stats.values())
    total_success = sum(len(s.successful) for s in all_stats.values())
    total_failures = total_requests - total_success

    overall_success_rate = total_success / total_requests * 100 if total_requests > 0 else 0

    print(f"Total requests:     {total_requests}")
    print(f"Successful:        {total_success} ({overall_success_rate:.1f}%)")
    print(f"Failed:            {total_failures}")
    print()

    if overall_success_rate >= 80:
        verdict = "GOOD"
    elif overall_success_rate >= 50:
        verdict = "MARGINAL"
    else:
        verdict = "POOR"
    print(f"Overall verdict:   {verdict}")

    engines_with_failures = [name for name, s in all_stats.items() if len(s.successful) < len(s.results) and len(s.successful) > 0]
    if engines_with_failures:
        print(f"Engines w/ partial failures: {engines_with_failures}")
    print()

    all_times = [r.total_ms for stats in all_stats.values() for r in stats.successful]
    if all_times:
        print(f"Global avg response:  {statistics.mean(all_times):.0f}ms ({statistics.mean(all_times)/1000:.1f}s)")
        print(f"Global p95 response: {statistics.quantiles(all_times, n=20)[18]:.0f}ms ({statistics.quantiles(all_times, n=20)[18]/1000:.1f}s)")
        print(f"Global max response:  {max(all_times):.0f}ms ({max(all_times)/1000:.1f}s)")

    print()
    print("=" * 80)
    print("STRESS TEST COMPLETE")
    print("=" * 80)

    with open("/tmp/stress_test_results.json", "w") as f:
        json.dump(
            {
                name: {
                    "total": len(stats.results),
                    "successful": len(stats.successful),
                    "success_rate": stats.success_rate,
                    "avg_ms": stats.avg_total_ms,
                    "median_ms": stats.median_total_ms,
                    "min_ms": stats.min_total_ms,
                    "max_ms": stats.max_total_ms,
                    "p95_ms": stats.p95_total_ms,
                    "stddev_ms": stats.stddev_total_ms,
                    "failures": [
                        {"prompt": r.prompt_index, "error": r.error[:200]}
                        for r in stats.results
                        if not r.success
                    ],
                    "per_request": [
                        {
                            "prompt": r.prompt_index,
                            "total_ms": r.total_ms,
                            "queue_wait_ms": r.queue_wait_ms,
                            "success": r.success,
                            "error": r.error[:200],
                            "model": r.model,
                        }
                        for r in stats.results
                    ],
                }
                for name, stats in all_stats.items()
            },
            f,
            indent=2,
        )
    print(f"Detailed results saved to /tmp/stress_test_results.json")


if __name__ == "__main__":
    asyncio.run(run_stress_test())