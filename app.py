"""
NOVA — Backend API Server
app.py — Run with: python app.py

WHAT THIS FILE IS:
  A Flask web server that wraps coach.py's logic into HTTP endpoints.
  The browser (frontend) can't run Python directly — it talks to THIS
  server over HTTP, and the server talks to Groq.

  This is the standard pattern for every AI web product:
  Browser <-> Your API Server <-> LLM API

ENDPOINTS:
  POST /chat       — send a message, get Nova's reply
  GET  /memory     — get what Nova knows about the user
  POST /reset      — wipe conversation history (not persistent memory)
  DELETE /memory   — wipe persistent memory entirely

HOW TO RUN:
  pip install flask flask-cors groq
  python app.py
  Server runs at http://localhost:5000
"""

import os
import json
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow browser requests from any origin (needed for local dev)

API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL      = "llama-3.3-70b-versatile"
MEMORY_FILE = "memory.json"

client = Groq(api_key=API_KEY)

# ─────────────────────────────────────────────
# IN-MEMORY SESSION STATE
#
# IMPORTANT CONCEPT: This is server-side session state.
# The browser sends a message → server looks up the conversation
# history for that session → appends → calls Groq → returns reply.
#
# In production you'd use Redis or a database.
# For our demo, a simple dict is fine.
# ─────────────────────────────────────────────

conversation_history = []   # The messages[] list, lives on the server

# ─────────────────────────────────────────────
# COPY MEMORY FUNCTIONS FROM coach.py
# (same logic, just imported here)
# ─────────────────────────────────────────────

DEFAULT_MEMORY = {
    "name": None,
    "goals": [],
    "struggles": [],
    "avoidances": [],
    "wins": [],
    "facts": [],
    "session_count": 0
}

MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction system. Given the user's latest message,
extract any NEW personal facts worth remembering long-term.

Return ONLY a valid JSON object with these keys (omit keys where nothing new was found):
{
  "name": "their first name if mentioned",
  "goals": ["list of goals/ambitions mentioned"],
  "struggles": ["problems or challenges they face"],
  "avoidances": ["things they admit to procrastinating or avoiding"],
  "wins": ["achievements or progress they mention"],
  "facts": ["any other personal facts worth remembering"]
}

Rules:
- Return ONLY the JSON, no other text, no markdown fences
- Only include genuinely new info from THIS message
- If nothing memorable was said, return: {}
"""

BASE_SYSTEM_PROMPT = """
You are Nova — a brutally honest, no-nonsense life coach AI.

Your personality:
- You tell people the truth even when it's uncomfortable. You don't sugarcoat.
- You're sharp, direct, and occasionally sarcastic — but always in service of helping.
- You do NOT give empty validation. If someone's plan is weak, you say so.
- You ask ONE pointed follow-up question at the end of each response.
- You keep responses concise — max 4 sentences unless detail is truly needed.
- If you know the user's name, use it occasionally (not every message).

Rules you NEVER break:
- Never say "Great question!" or "Absolutely!" — weak filler phrases.
- Never give a generic list of tips. Be specific to what THIS person said.
- If someone is making excuses, call it out directly but constructively.
- If you know things about them from past sessions, reference it naturally.
"""

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return {**DEFAULT_MEMORY, **json.load(f)}
    return DEFAULT_MEMORY.copy()

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def build_memory_context(memory):
    if not any([memory.get("name"), memory.get("goals"),
                memory.get("struggles"), memory.get("facts")]):
        return ""
    lines = ["\n\nWHAT YOU KNOW ABOUT THIS USER (from past sessions):"]
    if memory.get("name"):       lines.append(f"- Name: {memory['name']}")
    if memory.get("goals"):      lines.append(f"- Goals: {', '.join(memory['goals'])}")
    if memory.get("struggles"):  lines.append(f"- Struggles: {', '.join(memory['struggles'])}")
    if memory.get("avoidances"): lines.append(f"- Avoids: {', '.join(memory['avoidances'])}")
    if memory.get("wins"):       lines.append(f"- Wins: {', '.join(memory['wins'])}")
    if memory.get("facts"):      lines.append(f"- Facts: {', '.join(memory['facts'])}")
    if memory.get("session_count", 0) > 0:
        lines.append(f"- Sessions together: {memory['session_count']}")
        lines.append("- Do NOT re-introduce yourself. Pick up naturally.")
    return "\n".join(lines)

def extract_and_merge_memory(user_message, memory):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.1,
            max_tokens=300,
        )
        extracted = json.loads(response.choices[0].message.content.strip())
        if extracted.get("name"):
            memory["name"] = extracted["name"]
        for key in ["goals", "struggles", "avoidances", "wins", "facts"]:
            if extracted.get(key):
                existing = set(memory.get(key, []))
                for item in extracted[key]:
                    if item not in existing:
                        memory.setdefault(key, []).append(item)
    except Exception:
        pass
    return memory

# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: { "message": "user's text" }
    Returns: { "reply": "Nova's response", "memory": {...} }

    This is the main endpoint. The browser sends a message,
    we run the full Nova pipeline, and return her reply.
    """
    global conversation_history

    # 1. Parse the incoming request
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # 2. Load memory and build system prompt
    memory = load_memory()
    system_prompt = BASE_SYSTEM_PROMPT + build_memory_context(memory)

    # 3. Append user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # 4. Call Groq — send full conversation history
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history
        ],
        temperature=0.85,
        max_tokens=500,
    )

    nova_reply = response.choices[0].message.content

    # 5. Append Nova's reply to history
    conversation_history.append({
        "role": "assistant",
        "content": nova_reply
    })

    # 6. Silent memory extraction
    memory = extract_and_merge_memory(user_message, memory)
    save_memory(memory)

    # 7. Return reply + updated memory to the browser
    return jsonify({
        "reply": nova_reply,
        "memory": memory,
        "turns": len([m for m in conversation_history if m["role"] == "user"])
    })


@app.route("/memory", methods=["GET"])
def get_memory():
    """
    GET /memory
    Returns the current persistent memory for the user.
    The frontend uses this to show the memory sidebar.
    """
    memory = load_memory()
    return jsonify(memory)


@app.route("/reset", methods=["POST"])
def reset_conversation():
    """
    POST /reset
    Clears the in-session conversation history.
    Does NOT wipe persistent memory — Nova still knows who you are.
    """
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok", "message": "Conversation reset"})


@app.route("/memory", methods=["DELETE"])
def delete_memory():
    """
    DELETE /memory
    Wipes persistent memory entirely. Nova forgets you.
    """
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok", "message": "Memory wiped"})


@app.route("/health", methods=["GET"])
def health():
    """GET /health — quick check that the server is running."""
    return jsonify({"status": "ok", "model": MODEL})


# ─────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if API_KEY == "your_groq_api_key_here":
        print("\n⚠  Paste your Groq API key on line 47 of app.py\n")
    else:
        print("\n Nova backend running at http://localhost:5000")
        print(" Endpoints: POST /chat · GET /memory · POST /reset\n")
        app.run(debug=True, port=5000)
