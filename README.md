# AI Restaurant Recommendation Service

An AI-powered restaurant recommendation engine for Bangalore, built with FastAPI, Groq LLM, and sentence-transformers. Takes user preferences (location, price, rating, cuisine, free-text vibes) and returns ranked, LLM-explained restaurant recommendations from a 51,000+ restaurant dataset. Includes an analytics dashboard, A/B testing framework, user feedback loop, and caching layer.

## How It Works

```
User Preferences
       |
       v
  [Hard Filters]          -- location, price bucket, min rating, cuisine match
       |
       v
  [A/B Variant Assignment] -- randomly assign scoring weights variant
       |
       v
  [Cache Check]            -- return cached result if available (5min TTL)
       |
       v
  [Heuristic Scoring]     -- weighted rating + cuisine match + price alignment
       |
       v
  [Semantic Search]        -- sentence-transformers embeddings for free-text "vibes"
       |
       v
  [Groq LLM Re-ranking]   -- Llama 3.3 re-ranks top candidates + generates explanations
       |
       v
  [Analytics Recording]    -- track search, response time, filters used
       |
       v
  Ranked Recommendations with Explanations + Variant Tag
```

## Features

### Core Recommendation Engine
- **Smart Filtering** — Filter by 30 Bangalore areas, 4 price tiers, rating threshold, and 107 cuisine types
- **Semantic Search** — Free-text queries like "romantic rooftop dinner" influence results via embedding similarity
- **LLM Explanations** — Groq-powered Llama 3.3 re-ranks candidates and writes a short reason for each pick
- **Graceful Fallback** — If LLM fails, heuristic ranking is served without explanations

### Product & Growth Features
- **Analytics Dashboard** — Track search volume, response times, popular locations/cuisines, filter usage rates
- **User Feedback Loop** — Thumbs up/down on each recommendation; satisfaction rate tracked in analytics
- **A/B Testing Framework** — Experiment with different scoring weights; measure variant performance via feedback
- **Caching Layer** — TTL-based cache (5min) reduces repeat query latency; hit/miss rate visible in analytics
- **Shareable Links** — Copy a URL with search parameters pre-filled; recipient sees auto-populated results
- **Save/Bookmark** — Star favorite restaurants; persisted in session with dedicated Saved view

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data | Pandas (in-memory), Hugging Face Datasets |
| Frontend | Vanilla HTML/CSS/JS served by FastAPI |
| Testing | pytest (36 tests) |

## Project Structure

```
.
├── ARCHITECTURE.md                          # Detailed system design document
├── requirements.txt                         # Python dependencies
├── backend/
│   ├── app.py                               # FastAPI app (11 endpoints)
│   ├── static/
│   │   └── index.html                       # Frontend UI (Search + Analytics + Saved views)
│   ├── analytics/
│   │   ├── store.py                         # In-memory event store for search tracking
│   │   ├── aggregator.py                    # Computes summary stats from events
│   │   └── feedback.py                      # In-memory feedback store (thumbs up/down)
│   ├── ab_testing/
│   │   └── experiments.py                   # Experiment definitions, variant assignment, scoring weights
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
│   │   ├── retrieval.py                     # Filter → score → semantic → LLM → response pipeline
│   │   └── cache.py                         # Dict-based TTL cache with hit/miss tracking
│   └── tests/
│       ├── test_ingestion.py                # Dataset ingestion tests (1 test)
│       ├── test_recommendations.py          # API endpoint tests (10 tests)
│       ├── test_llm.py                      # Groq client tests with mocks (5 tests)
│       ├── test_embeddings.py               # Semantic search tests (4 tests)
│       ├── test_analytics.py                # Analytics tracking tests (4 tests)
│       ├── test_feedback.py                 # Feedback endpoint tests (4 tests)
│       ├── test_cache.py                    # Cache layer tests (3 tests)
│       └── test_ab_testing.py               # A/B testing tests (5 tests)
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
Returns available filter options (cities and cuisines).

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

**Response** includes `recommendations` (with score, reason, variant) and `total_candidates`.

### `GET /analytics`
Returns aggregated search analytics: total searches, avg response time, top locations, top cuisines, filter usage rates, cache stats, feedback summary.

### `POST /feedback`
Record user feedback (thumbs up/down) on a recommendation.
```json
{
  "restaurant_id": "42",
  "query_location": "BTM",
  "is_positive": true,
  "variant": "A"
}
```

### `GET /feedback/stats`
Returns feedback summary: total, positive, negative, satisfaction rate.

### `GET /cache/stats`
Returns cache performance: size, hits, misses, hit rate.

### `GET /ab-test/results`
Returns A/B experiment definition and per-variant satisfaction rates.

### `GET /share`
Serves the frontend with URL query params for shareable search links.

## Running Tests

```bash
pytest backend/tests/ -v
```

All 36 tests cover:
- Dataset ingestion pipeline
- API endpoint validation and filtering
- Groq LLM client (mocked — no API key needed)
- Semantic search embedding and scoring
- Analytics event tracking
- User feedback recording and stats
- Cache hit/miss behavior
- A/B variant assignment and weight differentiation

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
