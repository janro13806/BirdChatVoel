# main.py
import os
import json
import hashlib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from openai import OpenAI

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

CACHE_FILE = "openai_cache.json"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="BirdsChat API", version="1.0")

# In-memory cache, loaded on startup
cache: dict[str, str] = {}

class AskResponse(BaseModel):
    prompt: str
    answer: str
    cached: bool

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

def ask_with_cache(prompt: str) -> tuple[str, bool]:
    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key], True

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful ornithologist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=70
    )
    answer = response.choices[0].message.content.strip()
    cache[key] = answer
    return answer, False

@app.get("/ask", response_model=AskResponse)
def ask(prompt: str = Query(..., min_length=1, description="Bird description question")):
    """
    Ask about a bird. Returns a short, friendly description.
    """
    try:
        answer, cached = ask_with_cache(prompt)
        return AskResponse(prompt=prompt, answer=answer, cached=cached)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))