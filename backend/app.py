from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .ab_testing.experiments import (
    EXPERIMENTS,
    get_assignments,
    get_variant_stats,
    record_variant_feedback,
    record_variant_search,
    set_session_variant,
)
from .analytics.aggregator import compute_analytics
from .analytics.feedback import get_feedback, record_feedback
from .analytics.store import get_events, record_event
from .auth.dependencies import require_admin, require_user
from .auth.users import authenticate
from .recommendations.cache import get_cache_stats
from .recommendations.data_store import get_dataframe
from .recommendations.models import (
    FeedbackRequest,
    FeedbackResponse,
    LoginRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from .chat.intent import (
    CONFIDENCE_THRESHOLD,
    MAX_CLARIFICATIONS,
    accumulate_intent,
    extract_intent,
    generate_clarification,
    map_intent_to_request,
    update_conversation_state,
)
from .chat.models import (
    ChatRequest,
    ChatResponse,
    ChatResponseType,
    ConversationState,
)
from .recommendations.retrieval import get_recommendations

app = FastAPI(title="Restaurant Recommendation API", version="2.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "foodie-ai-secret-change-in-production"),
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ── Public endpoints ─────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    df = get_dataframe()
    cities = sorted(df["city"].dropna().unique().tolist())
    cuisines: set[str] = set()
    for val in df["cuisines"].dropna():
        for c in str(val).split(","):
            c = c.strip()
            if c:
                cuisines.add(c)
    return {"cities": cities, "cuisines": sorted(cuisines)}


# ── Auth endpoints ───────────────────────────────────────────────────────


@app.post("/auth/login")
def login(body: LoginRequest, request: Request) -> dict:
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["user"] = user
    return {"status": "ok", "user": user}


@app.post("/auth/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"status": "logged_out"}


@app.get("/auth/me")
def auth_me(user: dict = Depends(require_user)) -> dict:
    return user


# ── User endpoints ───────────────────────────────────────────────────────


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(
    body: RecommendationRequest,
    request: Request,
    user: dict = Depends(require_user),
) -> RecommendationResponse:
    # Pass session-persisted variant into the context so retrieval.py
    # reuses it instead of assigning a new random variant.
    session_variant = request.session.get("ab_variant")
    set_session_variant(session_variant)

    response = get_recommendations(body)

    # Persist the variant in the session on first search
    if not session_variant and response.variant:
        request.session["ab_variant"] = response.variant

    # Track per-variant search count
    if response.variant:
        record_variant_search(response.variant)

    return response


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    body: FeedbackRequest,
    user: dict = Depends(require_user),
) -> FeedbackResponse:
    record_feedback(
        body.restaurant_id,
        body.query_location,
        body.is_positive,
        body.variant,
    )
    if body.variant:
        record_variant_feedback(body.variant, body.is_positive)
    return FeedbackResponse(status="recorded", total_feedback=len(get_feedback()))


# ── Chat endpoint ────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    user: dict = Depends(require_user),
) -> ChatResponse:
    # 1. Load conversation state from session
    try:
        raw_state = request.session.get("chat_state")
        conv_state = ConversationState(**raw_state) if raw_state else ConversationState()
    except Exception:
        conv_state = ConversationState()

    # 2. Extract intent from message
    intent = extract_intent(body.message, conv_state)

    # 3. Accumulate intent across turns
    accumulated = accumulate_intent(dict(conv_state.accumulated_intent), intent)
    conv_state.accumulated_intent = accumulated

    # 4. Clarification path: low confidence, not total failure, under max clarifications
    if (
        0.0 < intent.confidence < CONFIDENCE_THRESHOLD
        and conv_state.clarification_count < MAX_CLARIFICATIONS
    ):
        question = generate_clarification(intent)
        conv_state.clarification_count += 1

        # Save turns
        conv_state = update_conversation_state(
            conv_state, body.message, question, intent,
        )
        request.session["chat_state"] = conv_state.model_dump()

        return ChatResponse(
            type=ChatResponseType.clarification,
            message=question,
            results=None,
            parsed_intent=accumulated,
        )

    # 5. Map intent to recommendation request
    rec_request = map_intent_to_request(intent, body.message)

    # 6. Reuse A/B variant session logic
    session_variant = request.session.get("ab_variant")
    set_session_variant(session_variant)

    response = get_recommendations(rec_request)

    if not session_variant and response.variant:
        request.session["ab_variant"] = response.variant
    if response.variant:
        record_variant_search(response.variant)

    # 7. Record analytics event
    record_event("chat_search", {
        "message": body.message,
        "intent_confidence": intent.confidence,
        "location": intent.location,
        "cuisines": intent.cuisines,
        "results_count": len(response.recommendations),
    })

    # 8. Build conversational response
    count = len(response.recommendations)
    if count == 0:
        reply = "I couldn't find any restaurants matching that. Could you try a different area or cuisine?"
    elif intent.occasion:
        reply = f"Found {count} great spots for {intent.occasion}! Here are my top picks:"
    elif intent.mood:
        reply = f"Here are {count} {intent.mood} restaurants I'd recommend:"
    elif intent.location:
        reply = f"Found {count} restaurants in {intent.location} for you:"
    else:
        reply = f"Here are {count} restaurants you might enjoy:"

    # 9. Save updated conversation state
    result_ids = [r.restaurant.id for r in response.recommendations]
    conv_state = update_conversation_state(
        conv_state, body.message, reply, intent, result_ids,
    )
    request.session["chat_state"] = conv_state.model_dump()

    return ChatResponse(
        type=ChatResponseType.results,
        message=reply,
        results=response,
        parsed_intent=accumulated,
    )


# ── Admin endpoints ──────────────────────────────────────────────────────


@app.get("/analytics")
def analytics(user: dict = Depends(require_admin)) -> dict:
    return compute_analytics(get_events())


@app.get("/feedback/stats")
def feedback_stats(user: dict = Depends(require_admin)) -> dict:
    fb = get_feedback()
    positive = sum(1 for f in fb if f["is_positive"])
    negative = len(fb) - positive
    return {
        "total": len(fb),
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / len(fb) * 100, 1) if fb else 0.0,
    }


@app.get("/cache/stats")
def cache_stats(user: dict = Depends(require_admin)) -> dict:
    return get_cache_stats()


@app.get("/ab-test/results")
def ab_test_results(user: dict = Depends(require_admin)) -> dict:
    fb = get_feedback()
    variant_a = [f for f in fb if f.get("variant") == "A"]
    variant_b = [f for f in fb if f.get("variant") == "B"]
    a_pos = sum(1 for f in variant_a if f["is_positive"])
    b_pos = sum(1 for f in variant_b if f["is_positive"])

    rate_a = round(a_pos / len(variant_a) * 100, 1) if variant_a else 0.0
    rate_b = round(b_pos / len(variant_b) * 100, 1) if variant_b else 0.0

    diff = abs(rate_a - rate_b)
    winner = None
    if diff >= 5.0 and variant_a and variant_b:
        winner = "A" if rate_a > rate_b else "B"

    return {
        "experiment": EXPERIMENTS.get("scoring_weights", {}),
        "total_assignments": len(get_assignments()),
        "variant_stats": get_variant_stats(),
        "results": {
            "A": {
                "total_feedback": len(variant_a),
                "positive": a_pos,
                "satisfaction_rate": rate_a,
            },
            "B": {
                "total_feedback": len(variant_b),
                "positive": b_pos,
                "satisfaction_rate": rate_b,
            },
        },
        "winner": winner,
    }


# ── Static / share ──────────────────────────────────────────────────────


@app.get("/share")
def share():
    return FileResponse(str(_STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))
