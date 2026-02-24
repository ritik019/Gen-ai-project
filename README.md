# AI Restaurant Recommendation Service

An AI-powered restaurant recommendation engine for Bangalore, built with FastAPI, Groq LLM, and sentence-transformers. Takes user preferences (location, price, rating, cuisine, free-text vibes) and returns ranked, LLM-explained restaurant recommendations from a 51,000+ restaurant dataset.

## How It Works

```
User Preferences
       |
       v
  [Hard Filters]          -- location, price bucket, min rating, cuisine match
       |
       v
  [Heuristic Scoring]     -- 60% rating + 30% cuisine match + 10% price alignment
       |
       v
  [Semantic Search]        -- sentence-transformers embeddings for free-text "vibes"
       |
       v
  [Groq LLM Re-ranking]   -- Llama 3.3 re-ranks top candidates + generates explanations
       |
       v
  Ranked Recommendations with Explanations
```

## Features

- **Smart Filtering** — Filter by 30 Bangalore areas, 4 price tiers, rating threshold, and 107 cuisine types
- **Semantic Search** — Free-text queries like "romantic rooftop dinner" influence results via embedding similarity
- **LLM Explanations** — Groq-powered Llama 3.3 re-ranks candidates and writes a short reason for each pick
- **Graceful Fallback** — If LLM fails, heuristic ranking is served without explanations
- **Web UI** — Clean, responsive single-page interface with live filters and result cards

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data | Pandas (in-memory), Hugging Face Datasets |
| Frontend | Vanilla HTML/CSS/JS served by FastAPI |
| Testing | pytest |

## Project Structure

```
.
├── ARCHITECTURE.md                          # Detailed system design document
├── requirements.txt                         # Python dependencies
├── backend/
│   ├── app.py                               # FastAPI app (endpoints: /, /health, /metadata, /recommendations)
│   ├── static/
│   │   └── index.html                       # Frontend UI
│   ├── data_ingestion/
│   │   ├── config.py                        # Ingestion config (dataset name, paths)
│   │   └── ingest.py                        # Downloads Zomato dataset from HuggingFace, normalizes to canonical schema
│   ├── embeddings/
│   │   ├── config.py                        # Embedding model config (all-MiniLM-L6-v2, 384-dim)
│   │   ├── encoder.py                       # Loads model, encodes text to vectors
│   │   └── precompute.py                    # Offline script to embed all 51k restaurants
│   ├── llm/
│   │   ├── config.py                        # Groq API config (key, model, timeout, feature flag)
│   │   └── groq_client.py                   # Builds prompt, calls Groq, parses JSON response
│   ├── recommendations/
│   │   ├── models.py                        # Pydantic request/response schemas
│   │   ├── data_store.py                    # Loads CSV + embeddings into memory (singleton)
│   │   └── retrieval.py                     # Filter → score → semantic → LLM → response pipeline
│   └── tests/
│       ├── test_ingestion.py                # Dataset ingestion tests
│       ├── test_recommendations.py          # API endpoint tests (10 tests)
│       ├── test_llm.py                      # Groq client tests with mocks (5 tests)
│       └── test_embeddings.py               # Semantic search tests (4 tests)
└── backend/data/processed/                  # Generated data (gitignored)
    ├── restaurants.csv                      # 51,717 normalized restaurants
    └── embeddings.npy                       # Precomputed embedding vectors (76MB)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Ingest the dataset

Downloads the Zomato dataset from Hugging Face and normalizes it into a canonical schema:

```bash
python -m backend.data_ingestion.ingest
```

This creates `backend/data/processed/restaurants.csv` with 51,717 restaurants.

### 4. Precompute embeddings

Embeds all restaurants using sentence-transformers for semantic search:

```bash
python -m backend.embeddings.precompute
```

This creates `backend/data/processed/embeddings.npy` (~76MB).

### 5. Run the server

```bash
uvicorn backend.app:app --reload
```

Open **http://localhost:8000** in your browser.

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok"}`.

### `GET /metadata`
Returns available filter options:
```json
{
  "cities": ["BTM", "Banashankari", "Koramangala 5th Block", ...],
  "cuisines": ["Afghan", "American", "Biryani", "Cafe", "Chinese", ...]
}
```

### `POST /recommendations`

**Request body:**
```json
{
  "location": "Koramangala",
  "price_range": ["$", "$$"],
  "min_rating": 3.5,
  "cuisines": ["Italian", "Cafe"],
  "free_text_preferences": "romantic date night",
  "limit": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `location` | string | Yes | City area or locality (e.g., "BTM", "Indiranagar") |
| `price_range` | string[] | No | Price buckets: `$`, `$$`, `$$$`, `$$$$` |
| `min_rating` | float | No | Minimum rating 0-5 (default: 0) |
| `cuisines` | string[] | No | Filter by cuisine types |
| `free_text_preferences` | string | No | Free-text for semantic search (e.g., "cozy rooftop") |
| `limit` | int | No | Number of results 1-50 (default: 10) |

**Response:**
```json
{
  "recommendations": [
    {
      "restaurant": {
        "id": "42",
        "name": "Cafe Azzure",
        "address": "3rd Floor, 80 Feet Road, Koramangala",
        "city": "Koramangala 5th Block",
        "locality": "Koramangala 5th Block",
        "price_bucket": "$$",
        "avg_cost_for_two": 600.0,
        "avg_rating": 4.3,
        "cuisines": ["Cafe", "Italian", "Continental"]
      },
      "score": 0.8234,
      "reason": "A cozy Italian cafe perfect for a romantic evening with great continental options."
    }
  ],
  "total_candidates": 127
}
```

## Running Tests

```bash
pytest backend/tests/ -v
```

All 20 tests cover:
- Dataset ingestion pipeline
- API endpoint validation and filtering
- Groq LLM client (mocked — no API key needed)
- Semantic search embedding and scoring

## Dataset

Uses the [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) dataset from Hugging Face.

| Stat | Value |
|------|-------|
| Total restaurants | 51,717 |
| City areas | 30 (all Bangalore) |
| Localities | 93 |
| Cuisine types | 107 |
| Price buckets | 4 ($, $$, $$$, $$$$) |
| Rating range | 1.8 - 5.0 |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, data model, request-response flow, LLM prompt strategy, and phase-wise development plan.
