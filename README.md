# Conversational E-Commerce Search Agent

A lightweight shopping-agent implementation for the TechJam Conversational E-Commerce Search Challenge. The agent searches a frozen catalog of 50,000 clothing, shoes, and jewelry products and returns ranked product recommendations using local SQLite FTS5 full-text search with BM25 ranking and additional filter algorithms.

The system runs fully locally without requiring an LLM or API key. Optionally, users can provide an LLM API key to enable enhanced intent understanding and recommendation quality, with the local pipeline retained as a reliable offline fallback.

## Project Overview

The challenge requires an `Agent` to identify a customer's hidden target product within a maximum of 10 conversational turns. On each turn, the evaluator provides a session ID, an anonymized user profile, the customer's message, and the requested recommendation count. The agent returns:

- A customer-facing response message
- A structured clarification attribute
- An ordered list of catalog `parent_asin` recommendations
- Optional token-usage information

### Retrieval pipeline

1. The product catalog is loaded into an in-memory SQLite FTS5 index using fields such as title, categories, features, details, store, and description.
2. User requirements are extracted and stored in a multi-turn SessionState, allowing preferences and constraints to accumulate or be updated throughout the conversation.
3. The current requirements are converted into a weighted FTS5 query, and BM25 retrieves and ranks relevant products.
4. Structured constraints, such as budget and product attributes, are used to further refine the candidate results.
5. When additional information would improve retrieval, the agent selects a relevant ask_attribute and asks the user a targeted clarification question.
6. **LLM usage is optional.** If an API key is provided, an LLM can assist with understanding complex user requirements and intent changes. Without one, the agent falls back to the fully local rule-based pipeline and remains functional without network access.

## Repository Structure

```text
.
├── data/
│   ├── public_set.jsonl            # 200 labeled development sessions
│   └── catalog.jsonl               # Frozen 50,000-product catalog
├── docs/
│   ├── agent_api_contract.json     # Machine-readable API contract
│   ├── baseline_results.json       # Reference evaluation results
│   └── evaluation_config.json      # Official evaluation settings
├── agent.py                        # Shopping agent implementation
├── local_evaluator.py              # Public-set evaluator
├── requirements.txt                # Python dependencies
├── results.json                    # Performance output metrics of agent
└── README.md
```

## Requirements

- Python 3.10 or later
- SQLite with FTS5 support
- `python-dotenv`
- `openai` (optional, only required for the OpenAI-compatible/Groq LLM fallback)

## Setup and Installation

Install the dependencies

```bash
pip install -r requirements.txt
```

Clone the repository and enter its directory:

```bash
git clone https://github.com/shzh05/Shopping-Copilot-TechJam-2026-tiki-tiki.git
cd Shopping-Copilot-TechJam-2026-tiki-tiki
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

Verify the catalog against published SHA256 checksum before running an evaluation.

## Reproducing the Results

Run the local evaluator from the repository root:

```bash
python3 -m evaluator.local_evaluator
```

The evaluator runs the agent on the 200 public sessions and reports Hit Rate@10, MRR, MTTC, Efficiency, and the combined technical score. It writes per-session results to `results.json`.

## Limitations and Future Improvements

- **Runtime:** Evaluating many multi-turn sessions can take several minutes because each turn performs intent extraction, BM25 retrieval, reranking, and clarification analysis. We would improve speed through stronger caching, optimized SQLite queries, and a persistent pre-built search index.

- **Rule-based extraction:** The system may struggle with synonyms, misspellings, implicit preferences, negations, and complex corrections. A future version could incorporate embeddings or a stronger language-understanding component.

- **Incomplete metadata:** Some catalog products lack prices, sizes, materials, or clearly structured categories. Additional catalog cleaning and attribute normalization would improve ranking accuracy.

- **In-memory session state:** Conversation state is lost when the program restarts and is not designed for distributed use. A persistent database could support longer-lived sessions.

## Dataset and Attribution

The frozen catalog and public sessions are derived from the Amazon Reviews 2023 dataset provided by McAuley Lab, UC San Diego. The competition organizers prepare and freeze the participant-facing catalog and evaluation sessions. Refer to the competition documentation and the original dataset documentation for the applicable attribution and redistribution terms.

## Team Member Contributions

- **Han Zhi Heng, Shawn** — Implemented the product search and filtering pipeline, including BM25 retrieval, RRF ranking, and constraint-based reranking.

- **Ouh Zirui** — Implemented the optional LLM integration and intent override functionality for updating or replacing user requirements.

- **Leong Qi An** — Implemented user requirement extraction and classification.

- **Zhong Chengjie** — Developed the attribute-selection component for identifying the most useful `ask_attribute` to request from the user.

- **Tan Jun Yu Henry** — Analysed the dataset and provided insights that guided the project’s design and development; fine-tuned retrieval weights and produced the demonstration video.