# FoodieAI — AI-Powered Restaurant Discovery Engine

An intent-driven restaurant recommendation system for Bangalore that replaces traditional filter-based search with conversational AI. Users type natural language like *"romantic Italian dinner in Koramangala under ₹1500"* and get ranked, LLM-explained results. Built with FastAPI, Groq LLM (Llama 3.3 70B), and sentence-transformers on a 51,000+ restaurant dataset.

**Live Demo:** [Deployed on Vercel](https://gen-ai-project.vercel.app) | **Login:** `user/user123` or `admin/admin123`

## How It Works

```
User Message: "romantic Italian dinner in Koramangala under ₹1500"
       │
       ▼
  [Intent Extraction]      ── Groq LLM parses: location, cuisines, mood, price, occasion
       │
       ├─ Low confidence? ──→ [Clarification] → "Which area are you looking at?"
       │                         (multi-turn conversation state in session)
       ▼
  [Map Intent → Filters]   ── location, price buckets, cuisines, free-text preferences
       │
       ▼
  [Hard Filters]            ── location, price bucket, min rating, cuisine match
       │
       ▼
  [A/B Variant Assignment]  ── session-persistent variant (A: rating-heavy, B: balanced)
       │
       ▼
  [Cache Check]             ── SHA256-hashed query key, 5-min TTL
       │
       ▼
  [Heuristic Scoring]       ── weighted rating + cuisine match + price alignment
       │
       ▼
  [Semantic Search]          ── 384-dim sentence-transformer embeddings for "vibe" matching
       │
       ▼
  [Groq LLM Re-ranking]     ── Llama 3.3 re-ranks top candidates + writes explanations
       │
       ▼
  Ranked Recommendations with AI Explanations
```

## Features

### Conversational AI Chat Layer
- **Natural Language Search** — Type "quiet cafe with wifi near BTM" instead of filling filter forms
- **LLM Intent Extraction** — Groq extracts structured preferences (location, cuisines, mood, occasion, price, vibe) from free-text
- **Confidence-Gated Clarification** — Vague queries trigger a follow-up question; specific queries skip straight to results
- **Multi-Turn Conversation** — State accumulates across messages (session cookie); "something romantic" → "Italian in Koramangala" merges both intents
- **Graceful Degradation** — If LLM fails, raw message flows into existing pipeline as free-text preferences

### Core Recommendation Engine
- **Smart Filtering** — 30 Bangalore areas, 4 price tiers, 107 cuisine types, rating threshold
- **Semantic Search** — Free-text queries influence results via embedding cosine similarity (50% heuristic + 50% semantic)
- **LLM Explanations** — Groq re-ranks candidates and writes a one-sentence reason for each pick
- **Price Sentiment Mapping** — "cheap" → $, "under 500 per person" → $/$$/$$$ via regex + keyword matching

### Product & Growth Features
- **A/B Testing Framework** — Session-persistent variant assignment; per-variant satisfaction tracking; winner detection
- **Admin Analytics Dashboard** — Chart.js visualizations: top locations, cuisines, price distribution, feedback breakdown
- **User Feedback Loop** — Thumbs up/down per recommendation; satisfaction rate tracked per variant
- **Caching Layer** — SHA256 TTL cache (5min); hit/miss rate in admin dashboard
- **Save & Share** — Heart-save restaurants to local storage; shareable search URLs

### Authentication & Authorization
- **Session-based Auth** — Secure cookie sessions (Starlette SessionMiddleware)
- **Role-based Access** — User sees Search + Saved; Admin unlocks Analytics dashboard
- **Bcrypt Password Hashing** — Passwords stored as bcrypt hashes
- **Route Protection** — 401 (unauthenticated) and 403 (insufficient role) enforcement

### Frontend (Mobile-First)
- **Warm Food-Inspired Theme** — Terracotta/burgundy/golden palette with Playfair Display + DM Sans fonts
- **Hero Conversational Search** — "What are you craving tonight?" with occasion quick-select pills (Date Night, Birthday, Work Cafe, etc.)
- **Restaurant Cards** — Cuisine-themed gradient headers, emoji overlays, star ratings, circular match score, AI insight chat bubbles
- **Collapsible Filters** — Apply/Clear buttons, active filter tags, chip-based selection with search
- **Skeleton Loading** — Shimmer animations during API calls
- **Responsive Breakpoints** — 480px, 768px, 1400px with horizontally-scrolling pills and bottom-sheet filters on mobile

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Auth | bcrypt + Starlette SessionMiddleware |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Data | Pandas (in-memory), Hugging Face Datasets |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Deployment | Vercel (serverless) |
| Testing | pytest (74 tests) |

## Project Structure

```
├── backend/
│   ├── app.py                          # FastAPI app (15 endpoints)
│   ├── static/index.html               # Frontend (Login + Search + Saved + Analytics)
│   ├── auth/
│   │   ├── users.py                    # User store with bcrypt
│   │   └── dependencies.py             # require_user(), require_admin()
│   ├── chat/                           # NEW — Conversational AI layer
│   │   ├── models.py                   # ChatRequest, ChatResponse, ExtractedIntent, ConversationState
│   │   └── intent.py                   # Intent extraction, clarification, price mapping, accumulation
│   ├── analytics/
│   │   ├── store.py                    # Event store
│   │   ├── aggregator.py              # Summary stats
│   │   └── feedback.py                # Feedback store
│   ├── ab_testing/
│   │   └── experiments.py             # Variant assignment, winner detection
│   ├── data_ingestion/
│   │   └── ingest.py                  # Downloads Zomato dataset from HuggingFace
│   ├── embeddings/
│   │   ├── encoder.py                 # Text → 384-dim vectors
│   │   └── precompute.py             # Offline embedding script
│   ├── llm/
│   │   ├── config.py                  # Groq API config
│   │   └── groq_client.py            # LLM re-ranking + explanations
│   ├── recommendations/
│   │   ├── models.py                  # Pydantic schemas
│   │   ├── data_store.py             # CSV + embeddings loader
│   │   ├── retrieval.py              # Filter → score → semantic → LLM pipeline
│   │   └── cache.py                  # TTL cache with hit/miss tracking
│   └── tests/                         # 74 tests
│       ├── test_chat.py               # Chat layer tests (22 tests)
│       ├── test_auth.py               # Auth/role tests (16 tests)
│       ├── test_recommendations.py    # API tests (10 tests)
│       ├── test_llm.py               # Groq client tests (5 tests)
│       ├── test_ab_testing.py         # A/B testing tests (5 tests)
│       ├── test_embeddings.py         # Semantic search tests (4 tests)
│       ├── test_analytics.py          # Analytics tests (4 tests)
│       ├── test_feedback.py           # Feedback tests (4 tests)
│       ├── test_cache.py             # Cache tests (3 tests)
│       └── test_ingestion.py          # Ingestion test (1 test)
└── backend/data/processed/            # Generated (gitignored)
    ├── restaurants.csv                # 51,717 restaurants
    └── embeddings.npy                 # Precomputed vectors (76MB)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment (.env file in project root)
GROQ_API_KEY=your_groq_api_key_here  # Free at console.groq.com

# 3. Ingest dataset
python -m backend.data_ingestion.ingest

# 4. Precompute embeddings
python -m backend.embeddings.precompute

# 5. Run server
uvicorn backend.app:app --reload
```

Open **http://localhost:8000** — login with `user/user123` or `admin/admin123`.

## API Reference

### Public
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metadata` | Available locations and cuisines |

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Authenticate and create session |
| POST | `/auth/logout` | Clear session |
| GET | `/auth/me` | Current user info |

### User (requires login)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/recommendations` | Filter-based recommendations |
| POST | `/chat` | **Conversational AI search** |
| POST | `/feedback` | Thumbs up/down feedback |

### Admin
| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics` | Search analytics data |
| GET | `/feedback/stats` | Feedback summary |
| GET | `/cache/stats` | Cache performance |
| GET | `/ab-test/results` | A/B experiment results |

### `POST /chat` (NEW)
```json
{ "message": "romantic Italian dinner in Koramangala under 1500" }
```
Returns `type: "results"` with recommendations, or `type: "clarification"` with a follow-up question.

### `POST /recommendations`
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

## Running Tests

```bash
pytest backend/tests/ -v
```

**74 tests** covering: authentication, chat layer, recommendations, LLM client, semantic search, analytics, feedback, cache, A/B testing, and data ingestion.

## Dataset

[ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) from Hugging Face.

| Stat | Value |
|------|-------|
| Total restaurants | 51,717 |
| City areas | 30 (Bangalore) |
| Cuisine types | 107 |
| Price buckets | 4 ($, $$, $$$, $$$$) |
