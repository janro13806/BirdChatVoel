import os
import json
from openai import OpenAI
import hashlib

# Initialize OpenAI client
client = OpenAI(api_key="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")

CACHE_FILE = "openai_cache.json"

# Load or initialize cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

def ask_with_cache(prompt: str) -> str:
    # Create a unique key for the prompt
    key = hashlib.sha256(prompt.encode()).hexdigest()

    if key in cache:
        print("✅ Using cached response")
        return cache[key]

    print("⚡ Calling OpenAI API...")
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

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    return answer

if __name__ == "__main__":
    bird = "Kingfisher"
    prompt = f"Give me a short, friendly description of the bird species {bird}."
    print(ask_with_cache(prompt))
