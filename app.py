from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import requests
import os
import uuid
import time
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth

app = Flask(__name__)

# -------------------------------------------------------
# CORS
# -------------------------------------------------------

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://eightfoldai-chat.netlify.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
# -------------------------------------------------------
# INIT FIREBASE
# -------------------------------------------------------
try:
    cred_json = os.getenv("FIREBASE_CREDENTIALS", "{}")
    if cred_json != "{}":
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin initialized")
    else:
        print("⚠️ Firebase not initialized — missing FIREBASE_CREDENTIALS")
except Exception as e:
    print("❌ Firebase init error:", str(e))

# -------------------------------------------------------
# Anthropic (Claude) CONFIG
# -------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"   # ✨ Your choice
ANTHROPIC_VERSION = "2023-06-01"               # REQUIRED

# -------------------------------------------------------
# ElevenLabs CONFIG
# -------------------------------------------------------
ELEVEN_KEYS = [k.strip() for k in os.getenv("ELEVEN_KEYS", "").split(",") if k.strip()]
VOICE_MAP = {
    "male": os.getenv("ELEVEN_VOICE_MALE", "pNInz6obpgDQGcFmaJgB"),
    "female": os.getenv("ELEVEN_VOICE_FEMALE", "21m00Tcm4TlvDq8ikWAM")
}

key_indices = {"eleven": 0}

# Session store
sessions = {}
user_sessions = {}


# -------------------------------------------------------
# AUTH MIDDLEWARE
# -------------------------------------------------------
def verify_firebase_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = auth_header.split("Bearer ")[1]

        try:
            decoded = auth.verify_id_token(token)
            request.user_id = decoded["uid"]
            request.user_email = decoded.get("email", "")
            return f(*args, **kwargs)
        except Exception as e:
            print("❌ Invalid Firebase token:", str(e))
            return jsonify({"error": "Invalid token"}), 401

    return wrapper


# -------------------------------------------------------
# HELPER — ElevenLabs rotation
# -------------------------------------------------------
def get_next_eleven_key():
    if not ELEVEN_KEYS:
        return None
    idx = key_indices["eleven"]
    key = ELEVEN_KEYS[idx]
    key_indices["eleven"] = (idx + 1) % len(ELEVEN_KEYS)
    return key


# -------------------------------------------------------
# HELPER — Claude Messages API wrapper
# -------------------------------------------------------
def call_claude(system_prompt, conversation):
    """
    system_prompt: str
    conversation: list of dict [{role:"user"/"assistant", content:""}]
    """

    if not ANTHROPIC_API_KEY:
        return {"error": "Claude API key missing"}

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json"
    }

    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 600,
        "system": system_prompt,
        "messages": conversation
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=45)

        if resp.status_code != 200:
            print("❌ Claude Error:", resp.status_code, resp.text)
            return {"error": resp.text}

        data = resp.json()

        # Claude Messages API returns:
        #   data["content"][0]["text"]
        text = data["content"][0]["text"]

        return {"text": text}

    except Exception as e:
        print("❌ Claude Exception:", str(e))
        return {"error": str(e)}


# -------------------------------------------------------
# SYSTEM PROMPT BUILDER
# -------------------------------------------------------
def create_system_prompt(domain, role, interview_type, difficulty):
    return f"""
You are a professional mock interview coach for:
- Domain: {domain}
- Role: {role}
- Type: {interview_type}
- Difficulty: {difficulty}

Rules:
1. Start with warm small talk.
2. Ask one interview question at a time.
3. Encourage the candidate.
4. Handle short answers, off-topic replies, and inappropriate content politely.
5. After 5–7 questions, produce a final summary with strengths, weaknesses, and score.

Output ALWAYS a JSON:
{{
 "text_response": "",
 "voice_response": "",
 "end": false
}}
"""


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.route("/")
def root():
    return {"status": "online", "claude_model": ANTHROPIC_MODEL}, 200


@app.route("/api/start-session", methods=["POST", "OPTIONS"])
@verify_firebase_token
def start_session():

    data = request.json or {}

    domain = data.get("domain")
    role = data.get("role")
    interview_type = data.get("interview_type", "Mixed")
    difficulty = data.get("difficulty", "Intermediate")
    duration = int(data.get("duration", 15))

    if not domain or not role:
        return {"error": "Missing domain/role"}, 400

    session_id = str(uuid.uuid4())
    system_prompt = create_system_prompt(domain, role, interview_type, difficulty)

    # FIRST Claude message
    conv = [
        {"role": "user", "content": "Start the interview with warm small talk."}
    ]

    result = call_claude(system_prompt, conv)

    if "error" in result:
        return {"error": result["error"]}, 500

    text = result["text"]

    # Save session
    sessions[session_id] = {
        "system_prompt": system_prompt,
        "messages": conv,
        "created_at": time.time(),
        "user_id": request.user_id
    }

    return {
        "session_id": session_id,
        "first_question": {
            "text_response": text,
            "voice_response": text
        }
    }, 200


@app.route("/api/chat", methods=["POST", "OPTIONS"])
@verify_firebase_token
def chat():

    data = request.json or {}
    session_id = data.get("session_id")
    user_msg = data.get("user_message")
    voice_style = data.get("voice_style", "male")

    if session_id not in sessions:
        return {"error": "Invalid session"}, 404

    session = sessions[session_id]

    # Build conversation
    conv = session["messages"]

    conv.append({"role": "user", "content": user_msg})

    result = call_claude(session["system_prompt"], conv)

    if "error" in result:
        return {"error": result["error"]}, 500

    text = result["text"]

    conv.append({"role": "assistant", "content": text})

    # Detect interview end
    end_flag = (
        "Thank you for completing this interview" in text or
        "[END" in user_msg
    )

    return {
        "text_response": text,
        "voice_response": text,
        "end": end_flag
    }, 200


@app.route("/api/tts", methods=["POST", "OPTIONS"])
@verify_firebase_token
def tts():
    data = request.json or {}
    text = data.get("text")
    voice_style = data.get("voice_style", "male")

    if not text:
        return {"error": "No text"}, 400

    api_key = get_next_eleven_key()
    if not api_key:
        return {"error": "Missing ElevenLabs key"}, 500

    voice_id = VOICE_MAP.get(voice_style, VOICE_MAP["male"])

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2"
    }

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json=payload, headers=headers
    )

    if resp.status_code != 200:
        print("❌ TTS error:", resp.text)
        return {"error": "TTS failed"}, 500

    return Response(resp.content, mimetype="audio/mpeg")

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers

        headers['Access-Control-Allow-Origin'] = request.headers.get('Origin')
        headers['Access-Control-Allow-Headers'] = "Content-Type, Authorization"
        headers['Access-Control-Allow-Methods'] = "GET, POST, OPTIONS"
        return response

# -------------------------------------------------------
# START SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("🚀 Claude Sonnet 4 backend running on port", port)
    app.run(host="0.0.0.0", port=port, debug=False)
