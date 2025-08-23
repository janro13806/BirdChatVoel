# main.py
import os
import json
import hashlib
from typing import Dict, Tuple, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ── Env ────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# Stronger small model + configurable via env
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CACHE_FILE = "openai_cache.json"

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="BirdsChat API", version="2.0")

# ── Cache ─────────────────────────────────────────────────────────────────────
cache: Dict[str, str] = {}   # stores raw JSON string per key


# ── Pydantic models for strict JSON shape ─────────────────────────────────────
class SimilarSpecies(BaseModel):
    name: str = Field(..., description="Species name")
    how_to_tell_apart: str = Field(..., description="1-line difference vs target")

class AskPayload(BaseModel):
    habitat_range: str
    voice: str
    similar_species: List[SimilarSpecies]
    quick_checklist: List[str]
    next_step: str

class AskResponse(BaseModel):
    prompt: str
    data: AskPayload
    cached: bool


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse, status_code=200)
async def root_health_check():
    return "OK"


@app.on_event("startup")
def load_cache():
    global cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = {}


@app.on_event("shutdown")
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ── System prompt: short, bullet-list teaching; JSON-only ─────────────────────
SYSTEM_PROMPT = """
You are an expert ornithologist and field instructor.

Return ONLY JSON that matches this schema (no prose, no markdown):

{
  "habitat_range": "string - region + typical habitats; note seasonality if relevant",
  "voice": "string - plain-language call/song description; short",
  "similar_species": [
    {"name": "string", "how_to_tell_apart": "string - 1-line difference"}
  ],
  "quick_checklist": ["string", "string", "... 3–6 bullets the user can verify now"],
  "next_step": "string - one action to confirm the ID"
}

Rules:
- Use compact phrasing; no long sentences.
- Each checklist bullet ≤ 15 words.
- Be honest about uncertainty.
- If the prompt is ambiguous or not a bird, include one clarifying question INSIDE next_step.
- Output must be valid JSON (no trailing commas, code fences, or commentary).
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cache_key(prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "model": MODEL_NAME}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call_openai(prompt: str) -> str:
    """Returns the model's raw JSON string (no formatting)."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=700
    )
    return resp.choices[0].message.content.strip()


def ask_with_cache(prompt: str) -> Tuple[AskPayload, bool]:
    key = _cache_key(prompt)
    if key in cache:
        raw_json = cache[key]
        # Validate cached JSON against schema
        data = AskPayload.model_validate_json(raw_json)
        return data, True

    raw = _call_openai(prompt)

    # Ensure it's valid JSON and matches schema
    try:
        # First check that it parses
        parsed = json.loads(raw)
        # Then validate against our schema
        payload = AskPayload.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        # Provide a helpful error with the model's raw output for debugging
        raise HTTPException(
            status_code=502,
            detail=f"Model did not return valid JSON for schema. Error: {str(e)} | Raw: {raw}"
        )

    # Cache the exact JSON string we validated
    cache[key] = json.dumps(payload.model_dump(), separators=(",", ":"))
    return payload, False


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.get("/ask", response_model=AskResponse)
def ask(
    prompt: str = Query(..., min_length=1, description="Bird description or question")
):
    """
    Returns a concise, bullet-style, JSON-structured birding lesson.
    """
    try:
        data, cached = ask_with_cache(prompt)
        return AskResponse(prompt=prompt, data=data, cached=cached)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
