# AI Restaurant Recommendation Service – Architecture

**Goal**: Take user preferences (price, place, rating, cuisine), use an LLM + Zomato dataset from Hugging Face, and return clear, ranked restaurant recommendations via an API/UI.

**Dataset**: [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) (Hugging Face).

---

## 1. High-Level System Design

### Client Layer

- **Web UI / API Client**
  - Form to capture: location/place, price range, min rating, cuisines, free-text preferences (e.g., “rooftop, romantic”).
  - Sends structured JSON to backend recommendation API.
  - Displays recommendations with explanations and key attributes.

### Backend Layer

- **API Service**
  - HTTP API (e.g., `POST /recommendations`).
  - Input validation, auth (optional), rate limiting, logging.
- **Recommendation Orchestrator**
  - Orchestrates the flow per request:
    - Normalize/validate preferences.
    - Retrieve candidate restaurants from preprocessed data.
    - Call LLM to re-rank and explain.
    - Assemble final response.

### Intelligence Layer

- **Candidate Retrieval Engine**
  - Uses structured data (from Zomato dataset) with filters (location, price, rating, cuisine).
  - Optional: simple scoring (rating, popularity).
- **LLM Reasoning Module**
  - Calls chosen LLM (via API) to:
    - Re-rank candidate list.
    - Generate short explanations per restaurant.
    - Optionally suggest alternatives.
- **(Optional) Embedding / Semantic Search Module**
  - Encodes restaurants and/or queries to embeddings.
  - Supports “vibes-based” queries from free-text.

### Data Layer

- **Raw Dataset Storage**
  - Ingestion from Hugging Face dataset.
  - Stored in local files (CSV/Parquet) or a DB.
- **Processed Store**
  - Normalized restaurant table (canonical schema).
  - Indexes for efficient filtering (by city, price bucket, rating, cuisines).
  - Optional vector index for embeddings.
- **Analytics & Feedback**
  - Logs of queries, responses, and user feedback to improve rankings and prompts.

---

## 2. Data Architecture

### Ingestion & Normalization

- Download dataset `ManikaSaini/zomato-restaurant-recommendation`.
- Clean/standardize:
  - Names, address, city, locality.
  - Price: map to buckets (e.g., `$`, `$$`, `$$$`, `$$$$`).
  - Ratings: normalize to 0–5 scale.
  - Cuisines: normalized tags list.
- Deduplicate if necessary.

### Canonical Restaurant Model

- `id`
- `name`
- `address`
- `city`
- `locality`
- `price_bucket`
- `avg_cost_for_two`
- `avg_rating`
- `cuisines` (list or delimited string)
- Optional:
  - `popularity_score` (based on rating / review count).
  - `embedding_vector` (for semantic search).
  - `highlights` / tags (if derivable).

### Storage Choices

- **Option A (simple)**: Single local file (CSV/Parquet) + in-memory filtering (good for dev/demo).
- **Option B (scalable)**: Relational DB (e.g., PostgreSQL) with indexes:
  - Indexes on `city`, `price_bucket`, `avg_rating`, `cuisines`.
  - Optional geospatial support if lat/long data is added.
- **Vector Store (optional later phase)**:
  - For embeddings: `restaurant_id` → vector.

---

## 3. Request–Response Flow

1. **Client → API**
   - `POST /recommendations` with JSON:
     - `location` (city or locality).
     - `price_range` (e.g., `low`, `medium`, `high`).
     - `min_rating`.
     - `cuisines` (list).
     - `free_text_preferences` (e.g., “date night, rooftop, live music”).
     - `limit` (number of results).

2. **API Validation & Normalization**
   - Validate ranges and enums.
   - Map `price_range` to internal representation / buckets.
   - Normalize cuisine strings.
   - Build internal `Preference` object.

3. **Candidate Retrieval**
   - Apply hard filters:
     - Location match.
     - Price bucket within requested range.
     - Rating >= `min_rating`.
     - At least one desired cuisine.
   - Optional: apply simple scoring (rating desc, popularity, distance).
   - Limit to top N candidates (e.g., 30–50) for LLM step.

4. **LLM Re-ranking & Explanation**
   - Build compact prompt/context:
     - User preferences (structured + free text).
     - Candidate list (only key fields: name, city, price, rating, cuisines).
   - Call LLM:
     - Ask it to output structured list of restaurant IDs with scores and short reasons.
   - Parse LLM result and map back to candidates.

5. **Response Assembly**
   - Join candidate data + LLM ranking + explanations.
   - Return JSON:
     - `recommendations`: list of:
       - `restaurant` (canonical fields),
       - `score` (if provided),
       - `reason` (LLM explanation).
     - Optional debug info (only in non-prod).

6. **Client Rendering**
   - Show ranked list with:
     - Name, rating, price, cuisines, city/locality.
     - Short explanation (“Great for a cozy date night…”).

---

## 4. LLM Integration Design

### LLM Client Abstraction

- Simple client with:
  - `rank_and_explain(preferences, candidates, max_results) -> list of {id, reason, score?}`.
- Pluggable implementation (e.g., Groq, OpenAI, others).

### Prompt Strategy (conceptual)

- System prompt: “You are a restaurant recommendation engine…”
- Provide:
  - Structured preferences.
  - Table/list of candidate restaurants with key attributes.
- Ask model to:
  - Respect hard constraints (city, price, min rating, cuisine).
  - Return JSON with:
    - Ordered recommendations (by id).
    - Short explanation per item.
- Guardrails:
  - Limit candidates and fields to stay within context.
  - Timeouts + graceful error handling and fallback.

### Fallback

- If LLM fails / times out / returns invalid JSON:
  - Use deterministic ranking only (no explanations).
  - Still return recommendations.

---

## 5. Phase-wise Plan

### Phase 1 – Foundations & Dataset Ingestion

- **Goals**
  - Initialize project structure.
  - Ingest and normalize Hugging Face dataset into canonical restaurant schema.

- **Key Work**
  - Project setup (language, framework, dependency management).
  - Ingestion script/module:
    - Download dataset.
    - Clean fields into canonical model.
    - Persist processed dataset (file or DB).
  - Basic documentation: data schema and ingestion steps.

### Phase 2 – Core Backend API & Deterministic Retrieval

- **Goals**
  - Provide a working API that returns reasonable recommendations using only filters and heuristics (no LLM yet).

- **Key Work**
  - API service (e.g., FastAPI/Express/etc.) with:
    - `POST /recommendations`.
  - Preference normalization/validation logic.
  - Candidate retrieval and heuristic ranking:
    - Filter by city, price, rating, cuisines.
    - Sort by rating, popularity, etc.
  - Basic tests for retrieval logic and API.

### Phase 3 – LLM Integration & Re-ranking

- **Goals**
  - Integrate LLM to re-rank candidates and generate explanations.

- **LLM Provider**: Groq (fast inference, OpenAI-compatible API).
  - Model: Llama 3 or Mixtral via Groq endpoint.
  - SDK: `groq` Python package.

- **Key Work**
  - Groq LLM client (using `groq` SDK).
  - Prompt builder and structured JSON response parser.
  - Orchestrator:
    - Network call to Groq with preferences + candidates.
    - Use LLM output to reorder and annotate recommendations.
  - Config:
    - `GROQ_API_KEY` environment variable.
    - Model name, timeouts, max tokens in config.
    - Feature flag to enable/disable LLM.

### Phase 4 – Embeddings & Semantic Search (Optional)

- **Goals**
  - Improve handling of free-text preferences and “vibes”.

- **Key Work**
  - Choose embedding model (could be same LLM family or separate).
  - Offline job:
    - Compute embeddings for restaurants (from names, cuisines, descriptions).
  - Online:
    - Embed user’s `free_text_preferences`.
    - Combine semantic similarity with hard filters.
  - Hybrid ranking logic: filter → semantic score → LLM re-rank (optional).

### Phase 5 – Frontend UI & UX

- **Goals**
  - Build a simple but polished UI for end-users.

- **Key Work**
  - Single-page web app:
    - Form with:
      - City/place selector.
      - Price slider or chips.
      - Rating slider.
      - Cuisine multi-select.
      - Free-text box for special preferences.
    - Results list with badges (price, rating, cuisines) and LLM explanations.
    - Loading/error states.
  - Integration with backend API.
  - Minimal styling and responsive layout.

### Phase 6 – Monitoring, Feedback & Iteration

- **Goals**
  - Measure quality and iterate on ranking and prompts.

- **Key Work**
  - Logging:
    - Requests (anonymized), responses, latencies, LLM usage.
  - Feedback:
    - Thumbs up/down or 1–5 rating per recommendation.
  - Evaluation:
    - Offline scripts/notebooks to inspect recommendations for various scenarios.
    - Prompt/ranking tweaks based on logs and feedback.
  - Performance:
    - Caching for common queries.
    - Limit LLM usage with fallbacks.

### Phase 7 – Frontend UI

- **Goals**
  - Build a web UI page for end-users to interact with the recommendation service.

- **Key Work**
  - Single-page web app (React or plain HTML/JS):
    - Form with:
      - City/locality selector (dropdown or autocomplete).
      - Price range chips (`$`, `$$`, `$$$`, `$$$$`).
      - Minimum rating slider.
      - Cuisine multi-select.
      - Free-text box for special preferences.
    - Results section:
      - Ranked restaurant cards with name, rating, price, cuisines, locality.
      - LLM-generated explanation per recommendation.
      - Loading spinner and empty/error states.
  - Integration with `POST /recommendations` backend API.
  - Minimal, responsive styling.
