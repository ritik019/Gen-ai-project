# AI Restaurant Recommendation Service

An AI-powered restaurant recommendation engine for Bangalore, built with FastAPI, Groq LLM, and sentence-transformers. Takes user preferences (location, price, rating, cuisine, free-text vibes) and returns ranked, LLM-explained restaurant recommendations from a 51,000+ restaurant dataset. Features role-based authentication, an admin analytics dashboard with Chart.js, A/B testing with session persistence and winner detection, user feedback loop, and caching layer.

## How It Works

```
User Login (session cookie)
       |
       v
User Preferences
       |
       v
  [Hard Filters]          -- location, price bucket, min rating, cuisine match
       |
       v
  [A/B Variant Assignment] -- session-persistent variant (A or B)
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

### Authentication & Authorization
- **Session-based Auth** — Login/logout with secure cookie sessions (FastAPI + Starlette SessionMiddleware)
- **Role-based Access** — User role sees Search + Saved; Admin role unlocks Analytics dashboard
- **Bcrypt Password Hashing** — Passwords stored as bcrypt hashes, never plaintext
- **Route Protection** — Middleware enforces 401 (unauthenticated) and 403 (insufficient role) on protected endpoints
- **Demo Accounts** — Pre-seeded `user/user123` (standard) and `admin/admin123` (admin)

### Core Recommendation Engine
- **Smart Filtering** — Filter by 30 Bangalore areas, 4 price tiers, rating threshold, and 107 cuisine types
- **Semantic Search** — Free-text queries like "romantic rooftop dinner" influence results via embedding similarity
- **LLM Explanations** — Groq-powered Llama 3.3 re-ranks candidates and writes a short reason for each pick
- **Graceful Fallback** — If LLM fails, heuristic ranking is served without explanations

### Product & Growth Features
- **Admin Dashboard** — Chart.js visualizations: top locations, top cuisines, price distribution, feedback breakdown
- **A/B Testing Framework** — Session-persistent variant assignment; per-variant satisfaction tracking; automatic winner detection (>= 5% difference)
- **User Feedback Loop** — Thumbs up/down on each recommendation; satisfaction rate tracked per variant
- **Caching Layer** — TTL-based cache (5min) reduces repeat query latency; hit/miss rate visible in admin dashboard
- **Shareable Links** — Copy a URL with search parameters pre-filled; recipient sees auto-populated results

### A/B Testing Explained
The system runs a **scoring weights experiment** with two variants:
- **Variant A** (Rating-heavy): Rating weight = 0.6, cuisine = 0.25, price = 0.15
- **Variant B** (Balanced): Rating weight = 0.4, cuisine = 0.35, price = 0.25

Each user session is assigned one variant on their first search. The variant persists across all searches in that session. Feedback (thumbs up/down) is tracked per variant. When one variant's satisfaction rate exceeds the other by >= 5%, the admin dashboard shows a **"Winner" badge**.

### Frontend
- **Red/White SaaS Theme** — Clean, professional design with `#B71C1C` primary and white backgrounds
- **Login Page** — Username/password form with demo credentials hint
- **Mobile-First Responsive** — Hamburger navigation, collapsible filters, stacked cards on mobile
- **Chart.js Admin Dashboard** — Horizontal bars, doughnut charts, pie charts for analytics
- **Touch-Friendly** — 44px minimum tap targets throughout

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Auth | bcrypt + Starlette SessionMiddleware (itsdangerous) |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data | Pandas (in-memory), Hugging Face Datasets |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Testing | pytest (52 tests) |

## Project Structure

```
.
├── ARCHITECTURE.md                          # Detailed system design document
├── requirements.txt                         # Python dependencies
├── backend/
│   ├── app.py                               # FastAPI app (14 endpoints, SessionMiddleware)
│   ├── static/
│   │   └── index.html                       # Frontend UI (Login + Search + Saved + Analytics)
│   ├── auth/
│   │   ├── __init__.py                      # Package init
│   │   ├── users.py                         # In-memory user store with bcrypt, authenticate()
│   │   └── dependencies.py                  # require_user(), require_admin() — FastAPI Depends
│   ├── analytics/
│   │   ├── store.py                         # In-memory event store for search tracking
│   │   ├── aggregator.py                    # Computes summary stats from events
│   │   └── feedback.py                      # In-memory feedback store (thumbs up/down)
│   ├── ab_testing/
│   │   └── experiments.py                   # Variant assignment (session-persistent), winner detection
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
│   │   ├── models.py                        # Pydantic request/response schemas (+ LoginRequest)
│   │   ├── data_store.py                    # Loads CSV + embeddings into memory (singleton)
│   │   ├── retrieval.py                     # Filter → score → semantic → LLM → response pipeline
│   │   └── cache.py                         # Dict-based TTL cache with hit/miss tracking
│   └── tests/
│       ├── test_auth.py                     # Auth/login/role tests (16 tests)
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

Open **http://localhost:8000** in your browser. Log in with `user/user123` or `admin/admin123`.

## API Reference

### Authentication

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | Public | Authenticate and create session |
| POST | `/auth/logout` | Public | Clear session |
| GET | `/auth/me` | User | Return current user info |

**Login request:**
```json
{ "username": "user", "password": "user123" }
```

**Login response:**
```json
{ "status": "ok", "user": { "username": "user", "role": "user" } }
```

### Public Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metadata` | Available locations and cuisines |
| GET | `/` | Serve frontend |
| GET | `/share` | Shareable search link |

### Protected Endpoints (User)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/recommendations` | Get restaurant recommendations |
| POST | `/feedback` | Submit thumbs up/down feedback |

### Admin-Only Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics` | Search analytics dashboard data |
| GET | `/feedback/stats` | Feedback summary stats |
| GET | `/cache/stats` | Cache performance metrics |
| GET | `/ab-test/results` | A/B experiment results + winner |

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

### `POST /feedback`
```json
{
  "restaurant_id": "42",
  "query_location": "BTM",
  "is_positive": true,
  "variant": "A"
}
```

## Running Tests

```bash
pytest backend/tests/ -v
```

All 52 tests cover:
- **Authentication** — login/logout, role checks, route protection (401/403)
- **Dataset ingestion** — HuggingFace download and normalization
- **API endpoints** — filtering, scoring, response format
- **Groq LLM client** — prompt building, response parsing (mocked)
- **Semantic search** — embedding similarity ranking
- **Analytics** — event tracking and aggregation
- **Feedback** — recording and satisfaction stats
- **Cache** — hit/miss behavior and stats endpoint
- **A/B testing** — variant assignment, weight differentiation, results endpoint

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
