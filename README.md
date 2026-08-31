# Conversational E-Commerce Search Agent

A lightweight shopping-agent implementation for the TechJam Conversational E-Commerce Search Challenge. The agent searches a frozen catalog of 50,000 clothing, shoes, and jewelry products and returns ranked product recommendations using local SQLite FTS5 full-text search with BM25 ranking.

The implementation is designed to run locally without an LLM, external API, vector database, or API key. This makes the baseline inexpensive, reproducible, and suitable for environments where network access may be disabled during final evaluation.

## Project Overview

The challenge requires an `Agent` to identify a customer's hidden target product within a maximum of 10 conversational turns. On each turn, the evaluator provides a session ID, an anonymized user profile, the customer's message, and the requested recommendation count. The agent returns:

- A customer-facing response message
- An optional structured clarification attribute
- An ordered list of catalog `parent_asin` recommendations
- Optional token-usage information

### Retrieval pipeline

1. The catalog is loaded from `catalog.jsonl`.
2. Product fields including title, categories, features, details, store, and description are indexed in an in-memory SQLite FTS5 table.
3. The user message is tokenized, normalized to lowercase, and filtered using a small stopword list.
4. The remaining terms are converted into an OR-based FTS5 query.
5. SQLite BM25 ranks the matching products using field-specific weights, with the title receiving the greatest weight.
6. The highest-ranked valid `parent_asin` values are returned to the evaluator.

The current implementation is intentionally deterministic and stateless between turns. The anonymized user profile is accepted through the required interface but is not yet used for personalization.

## Repository Structure

```text
.
├── agent.py                  # Shopping agent implementation
├── local_evaluator.py        # Deterministic public-set evaluator
├── public_set.jsonl          # 200 labeled development sessions
├── catalog.jsonl             # Frozen 50,000-product catalog
├── evaluation_config.json    # Official evaluation settings
├── agent_api_contract.json   # Machine-readable API contract
├── baseline_results.json     # Reference evaluation results
└── README.md
```

The exact filenames may be placed into the organizer's recommended folders, such as `starter/`, `data/`, and `docs/`, provided that the paths in the commands below are updated accordingly.

## Requirements

- Python 3.10 or later
- SQLite with FTS5 support, normally included with standard Python installations
- No third-party Python packages
- No API keys or external service credentials

## Setup and Installation

Clone the repository and enter its directory:

```bash
git clone <YOUR_PUBLIC_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Place the provided `catalog.jsonl` and `public_set.jsonl` files in the paths expected by the code. The default agent constructor expects:

```text
data/catalog.jsonl
```

If the catalog is supplied as `catalog.jsonl.gz`, decompress it with:

```bash
gzip -dk catalog.jsonl.gz
mkdir -p data
mv catalog.jsonl data/catalog.jsonl
```

Verify the catalog against the organizer's published SHA256 checksum before running an evaluation.

## Reproducing the Results

Run the local evaluator from the repository root:

```bash
python3 local_evaluator.py
```

If the evaluator is organized as a Python module, use the organizer-provided command instead:

```bash
python3 -m evaluator.local_evaluator
```

The evaluator runs the agent on the 200 public sessions and reports Hit Rate@10, MRR, MTTC, Efficiency, and the combined technical score. It may also write per-session results to `results.json`.

### Reference baseline

The published weak BM25 baseline reports the following results on the public set:

| Metric | Result |
|---|---:|
| Sessions | 200 |
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| Technical score | 0.10671 |

The technical score is calculated as:

```text
0.50 × Hit Rate@10 + 0.30 × MRR + 0.20 × Efficiency
```

Only exact `parent_asin` matches count as successful recommendations.

## Agent API

The implementation exports the required interface:

```python
from agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset("example-session", user_profile={})

response = agent.respond(
    "example-session",
    "I am looking for comfortable black running shoes",
    turn=1,
    top_k=10,
)
```

A response has the following form:

```python
{
    "message": "Here are the closest matches I found.",
    "ask_attribute": None,
    "recommendations": [
        {"parent_asin": "B000..."}
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0
    }
}
```

## Limitations and Future Improvements

The current system is a strong reproducible retrieval baseline, but it does not yet implement the full potential of a conversational shopping agent.

- It does not maintain structured requirements across turns.
- It does not detect or explicitly handle Buying, Browsing, Boundary, or Intent Override scenarios.
- It does not ask adaptive clarification questions; `ask_attribute` is currently `null`.
- It does not use the anonymized user profile for personalization.
- OR-based keyword matching can retrieve noisy results when the query is vague or uses synonyms that do not occur in the catalog.
- It does not perform semantic/vector retrieval or a semantic reranking stage.
- Product price and other structured constraints are not explicitly parsed or filtered.
- Index construction occurs at startup and may require additional optimization for constrained hardware.

Given more time, we would add a persistent conversation-state tracker, intent and constraint extraction, adaptive question selection, structured filtering for price/category/size/color, hybrid lexical and semantic retrieval, profile-aware reranking, and scenario-specific evaluation analysis. We would also benchmark startup time, memory usage, latency, and retrieval quality separately for each scenario.

## Model, API, Cost, and Privacy Disclosure

- Model: No external LLM or trained model is required by the current implementation.
- APIs: No external APIs are used.
- Dependencies: Python standard library, including `sqlite3` and SQLite FTS5.
- Network access: Not required for agent execution or local evaluation.
- Estimated model cost: $0.
- Reported token usage: 0 prompt tokens and 0 completion tokens per response.
- Secrets: No API keys or credentials should be committed to this repository.

## Dataset and Attribution

The frozen catalog and public sessions are derived from the Amazon Reviews 2023 dataset provided by McAuley Lab, UC San Diego. The competition organizers prepare and freeze the participant-facing catalog and evaluation sessions. Refer to the competition documentation and the original dataset documentation for the applicable attribution and redistribution terms.

## Team Member Contributions

<!-- Add team member names and contributions here. -->

## License

Add the license applicable to your submitted code and competition data here before making the repository public.
