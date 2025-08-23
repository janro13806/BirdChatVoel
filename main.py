# main.py
import os
import json
import hashlib
from typing import Dict, Tuple, List, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CACHE_FILE = "openai_cache.json"

# Fixed textbox limits (edit these if needed)
DEFAULT_MAX_LINES = 3
DEFAULT_MAX_CHARS = 200

# ── Setup ─────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="BirdsChat API", version="2.2")
cache: Dict[str, str] = {}  # stores rendered 'answer' text per key


# ── Pydantic models (internal structure for quality) ──────────────────────────
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
    answer: str
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


# ── System prompt: JSON-only, concise, no placeholders ────────────────────────
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
  "next_step": "string - one action to confirm the ID (may include one clarifying question)"
}

Constraints:
- Never write placeholders like "N/A" or empty arrays/strings.
- If the prompt is ambiguous or not a bird name, produce GENERIC but useful birding guidance
  in all fields and include a clarifying question inside next_step.
- Be specific but brief; each checklist bullet ≤ 15 words.
- Output must be valid JSON (no trailing commas, code fences, or commentary).
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cache_key(prompt: str) -> str:
    # include limits so cache won't clash if you change constants later
    payload = json.dumps({
        "prompt": prompt,
        "model": MODEL_NAME,
        "lines": DEFAULT_MAX_LINES,
        "len": DEFAULT_MAX_CHARS
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _call_openai(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

def _normalize_payload(parsed: dict) -> dict:
    """Ensure no empty or placeholder fields; backfill generic guidance when needed."""
    def bad(x: Any) -> bool:
        return (
            x is None or
            (isinstance(x, str) and (not x.strip() or x.strip().lower() in {"n/a", "na", "unknown"})) or
            (isinstance(x, list) and len(x) == 0)
        )

    generic = {
        "habitat_range": "Urban gardens, parks, and wetlands in Southern Africa; many species year-round.",
        "voice": "Note rhythm and repetition; track pitch steps and phrase length.",
        "similar_species": [
            {"name": "Common Bulbul", "how_to_tell_apart": "Dark head, yellow vent; bubbly song."},
            {"name": "Cape Sparrow", "how_to_tell_apart": "Stout bill; chirped phrases; bold male head pattern."}
        ],
        "quick_checklist": [
            "Size vs dove or sparrow",
            "Bill shape: thin/stout/hooked",
            "Color blocks and wing bars",
            "Tail length and outer-tail pattern",
            "Behavior: hovering, tail-flicking, ground hopping"
        ],
        "next_step": "Is this a bird species? If unsure, photo bill/tail and record 10s of song."
    }

    if bad(parsed.get("habitat_range")):
        parsed["habitat_range"] = generic["habitat_range"]
    if bad(parsed.get("voice")):
        parsed["voice"] = generic["voice"]
    if bad(parsed.get("similar_species")):
        parsed["similar_species"] = generic["similar_species"]
    if bad(parsed.get("quick_checklist")):
        parsed["quick_checklist"] = generic["quick_checklist"]
    if bad(parsed.get("next_step")):
        parsed["next_step"] = generic["next_step"]

    return parsed

def _render_textbox(p: AskPayload) -> str:
    """
    Render into a short textbox string:
    - ≤ DEFAULT_MAX_LINES lines
    - ≤ DEFAULT_MAX_CHARS chars total (including newlines)
    - Never truncate a fact; if a fact doesn't fit, skip it entirely.
    """
    similar = "; ".join(f"{s.name}: {s.how_to_tell_apart}" for s in p.similar_species[:1])
    checklist = "; ".join(p.quick_checklist[:2])

    # Order = importance priority
    facts = [
        f"Habitat: {p.habitat_range}",
        f"Voice: {p.voice}",
        f"Similar: {similar}" if similar else None,
        f"Checklist: {checklist}" if checklist else None,
        f"Next: {p.next_step}",
    ]
    facts = [f for f in facts if f]

    out: list[str] = []
    remaining = DEFAULT_MAX_CHARS

    for fact in facts:
        if len(out) >= DEFAULT_MAX_LINES:
            break
        # account for newline if this is not the first line
        need = len(fact) + (1 if out else 0)
        if need <= remaining:
            out.append(fact)
            remaining -= need
        else:
            # Skip the entire fact if it can't fit; do not truncate or add ellipsis
            continue

    return "\n".join(out)


def ask_with_cache(prompt: str) -> Tuple[str, bool]:
    key = _cache_key(prompt)
    if key in cache:
        return cache[key], True

    raw = _call_openai(prompt)

    try:
        parsed = json.loads(raw)
        parsed = _normalize_payload(parsed)
        payload = AskPayload.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model did not return valid JSON for schema. Error: {str(e)} | Raw: {raw}"
        )

    answer = _render_textbox(payload)
    cache[key] = answer
    return answer, False


# ── Endpoint (prompt only; textbox output) ────────────────────────────────────
@app.get("/ask", response_model=AskResponse)
def ask(prompt: str = Query(..., min_length=1, description="Bird description or question")):
    try:
        answer, cached = ask_with_cache(prompt)
        return AskResponse(prompt=prompt, answer=answer, cached=cached)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
