# AEGIS

**An uncertainty-aware agent workflow for simulated UAV telemetry analysis.**

AEGIS is a portfolio and research prototype that explores how a graph-based AI
system can combine classification, local explanations, policy retrieval,
geographic context, evidence fusion, report generation, and human review.

It operates exclusively on synthetic data. It is not a weapon system, an
operational ISR product, or a validated safety system.

## Why this project exists

Most agent demos stop at a chat interface. AEGIS focuses on harder engineering
questions:

- How should uncertain model output be routed?
- Can every recommendation be traced to model, context, or retrieved evidence?
- Does the system distinguish a recommendation from a human-approved action?
- Are explanations faithful to the model being used?
- Can evaluation claims be reproduced from the repository?

## Implemented workflow

```text
telemetry
   │
   ▼
ingestion ──► classification ──► local attribution ──► doctrine retrieval
                                                        │
                                                        ▼
context reasoning ──► evidence fusion ──► review gate (conditional)
                                              │
                                              ▼
                                     report ──► escalation recommendation
```

The LangGraph route is conditional: conflicting, low-confidence, or degraded
runs pass through a human-review node. A pending review prevents the system
from claiming that an alert was dispatched.

### Components

- **Classifier:** gradient-boosted model over nine telemetry features and a
  MiniLM sentence embedding, with sigmoid probability calibration.
- **Explanation:** leave-one-feature-out local attribution. Each value is the
  change in predicted-class probability after replacing one input with a
  documented neutral reference. The system does not label fallback values as
  SHAP.
- **Retrieval:** ChromaDB semantic search over eight synthetic policy excerpts,
  including a relevance-quality retry.
- **Context:** deterministic restricted-zone and corridor calculations.
- **Fusion:** combines model and geographic risk while treating retrieval
  relevance as evidence quality—not threat evidence.
- **Reporting:** Groq-backed structured generation when configured; otherwise
  an immediate, clearly labelled deterministic evidence-only report.
- **Review:** explicit `PENDING` state for uncertainty, conflicts, or degraded
  execution.
- **Observability:** per-node duration and status in every API result.

## Quick start

Python 3.11 is recommended.

```bash
cd AEGIS
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python data/simulated/generate_telemetry.py
uvicorn api.main:app --reload
```

In another terminal:

```bash
cd AEGIS
streamlit run dashboard/app.py
```

Open:

- Dashboard: http://localhost:8501
- OpenAPI: http://localhost:8000/docs
- Readiness: http://localhost:8000/health

An LLM key is optional. Without `GROQ_API_KEY`, the report node uses the
deterministic fallback without making a network request.

### Docker

```bash
cd AEGIS
docker compose up --build
```

The dashboard uses `AEGIS_API_BASE`; Compose configures it to reach the API
service over the internal network.

## Evaluation

Run the auditable synthetic benchmark:

```bash
cd AEGIS
python evaluation/evaluate.py
```

This writes `evaluation/results.json` and reports:

- per-class precision, recall, and F1
- confusion matrices
- expected calibration error
- p50 and p95 classifier latency
- separate results for the generator-like demo set and a small stress set with
  overlapping signals and unseen language

The limitations are part of the output. In particular, the demo data resembles
the training generator and must not be treated as independent validation.

## Tests and quality

```bash
pytest -q --cov=.
ruff check .
```

Tests cover individual agents, graph routing, observability, deterministic
fallback behavior, review safety, health readiness, and API validation. GitHub
Actions runs generation, linting, and tests on every push and pull request.

## API example

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  --data @data/simulated/example_request.json
```

The response contains the classification, probability attribution, retrieved
policy reference, context and fusion scores, review status, recommendation,
node timings, pipeline version, and any degraded-component errors.

## Repository structure

```text
AEGIS/
├── agents/              # workflow nodes
├── api/                 # FastAPI schemas and routes
├── core/                # state, configuration, LangGraph orchestration
├── dashboard/           # Streamlit operator UI
├── data/simulated/      # generator and synthetic scenarios
├── evaluation/          # reproducible benchmark
├── models/              # training and model loading
├── tests/               # unit, integration, and API tests
├── Dockerfile
└── docker-compose.yml
```

## Known limitations

- All telemetry and policy documents are synthetic.
- There is no image or video input.
- The classifier is not validated on operational data.
- The stress benchmark is deliberately small.
- Policy retrieval is a demonstration corpus, not an authoritative source.
- The external LLM report path still requires broader faithfulness evaluation.
- Review is represented as workflow state; operator identity and durable
  approval storage are future work.
- Authentication, rate limiting, persistent audit storage, and production
  deployment hardening are not implemented.

## Roadmap

1. Temporal trajectory modelling instead of single telemetry packets.
2. Held-out generator families and adversarial distribution-shift tests.
3. Conformal prediction or abstention for calibrated uncertainty.
4. Durable LangGraph checkpoints and authenticated review decisions.
5. Citation-level report verification against retrieved evidence.
6. Optional image-track fusion with a separately evaluated vision model.

## License

[MIT](LICENSE)
