from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import requests
import os
import uuid
import time
import re
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
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_VERSION = "2023-06-01"

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
    Returns: dict with text_response, voice_response, end
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
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": conversation
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=45)

        if resp.status_code != 200:
            print("❌ Claude Error:", resp.status_code, resp.text)
            return {"error": f"Claude API error: {resp.status_code}"}

        data = resp.json()

        # Claude Messages API returns: data["content"][0]["text"]
        text = data["content"][0]["text"]
        
        # Try to parse as JSON first
        try:
            # Remove markdown code blocks if present
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(text)
            
            # Ensure required fields
            if 'text_response' not in parsed:
                parsed['text_response'] = text
            if 'voice_response' not in parsed:
                parsed['voice_response'] = parsed['text_response']
            if 'end' not in parsed:
                parsed['end'] = False
            
            # Clean voice_response (remove emojis, markdown)
            if parsed.get('voice_response'):
                voice = parsed['voice_response']
                # Remove emojis and special characters
                voice = re.sub(r'[^\x00-\x7F]+', '', voice)
                voice = voice.replace('*', '').replace('#', '').replace('_', '').replace('`', '')
                voice = ' '.join(voice.split())
                parsed['voice_response'] = voice
            
            print(f"✅ Claude Success - Parsed JSON")
            return parsed
            
        except json.JSONDecodeError:
            # If not JSON, try to extract JSON with regex
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if 'text_response' in parsed:
                        print(f"✅ Claude Success - Extracted JSON")
                        return parsed
                except:
                    pass
            
            # Fallback: return plain text as response
            print(f"⚠️ Claude returned non-JSON, using plain text")
            return {
                "text_response": text,
                "voice_response": re.sub(r'[^\x00-\x7F]+', '', text),
                "end": False
            }

    except Exception as e:
        print("❌ Claude Exception:", str(e))
        return {"error": str(e)}


# -------------------------------------------------------
# SYSTEM PROMPT BUILDER
# -------------------------------------------------------
def create_system_prompt(domain, role, interview_type, difficulty):
    return f"""You are "AI Interview Practitioner," a professional mock interview coach.

The user has selected:
- Domain: {domain}
- Role: {role}
- Interview Type: {interview_type}
- Difficulty: {difficulty}

INTERVIEW FLOW:
1. FIRST MESSAGE: Start with warm small talk (1-2 sentences). Ask how they're feeling about the interview.
2. SECOND MESSAGE: Acknowledge warmly and ask one more casual question.
3. THIRD MESSAGE: Transition: "Great! Let's dive into the interview. Here's my first question..." then ask first interview question.
4. CONTINUE: Ask ONE question at a time, be encouraging.
5. AFTER 5-7 QUESTIONS: Provide final summary.

EDGE CASES:
- Irrelevant answers: Gently redirect to interview focus
- Rude behavior: Maintain professionalism, may end early if severe
- Short answers: Ask for elaboration
- Gibberish: Ask to rephrase
- "[END_INTERVIEW_TIME_UP]" or "[END_INTERVIEW_MANUAL]": Generate summary immediately

CRITICAL: You MUST respond in valid JSON format ONLY. No extra text before or after.

{{
  "text_response": "Your message here (can include emojis)",
  "voice_response": "Same message but plain text only, no emojis or symbols",
  "end": false
}}

FINAL SUMMARY FORMAT (after 5-7 questions OR when ending):
{{
  "text_response": "Thank you for completing this interview! Here's your performance summary.",
  "voice_response": "Thank you for completing this interview! Here's your performance summary.",
  "strengths": "3-4 specific strengths with examples",
  "weaknesses": "2-3 areas needing improvement",
  "score": 85,
  "communication_score": 80,
  "technical_score": 85,
  "confidence_score": 90,
  "behavior_score": 85,
  "overall_impression": "2-3 sentences about overall performance",
  "recommendations": "3-4 actionable steps to improve",
  "selected": true,
  "end": true
}}

SELECTION CRITERIA:
- Selected (true): Score >= 65, answered reasonably, showed effort
- Not Selected (false): Score < 65, poor communication, inappropriate behavior

Remember: Output ONLY valid JSON, nothing else!"""


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.route("/")
def root():
    return {"status": "online", "model": ANTHROPIC_MODEL}, 200


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
        {"role": "user", "content": "Start the interview with warm small talk as instructed in your system prompt."}
    ]

    result = call_claude(system_prompt, conv)

    if "error" in result:
        return {"error": result["error"]}, 500

    # Save conversation
    conv.append({"role": "assistant", "content": result.get("text_response", "")})

    # Save session
    sessions[session_id] = {
        "system_prompt": system_prompt,
        "messages": conv,
        "created_at": time.time(),
        "user_id": request.user_id,
        "exchange_count": 0,
        "question_count": 0,
        "domain": domain,
        "role": role,
        "interview_type": interview_type,
        "difficulty": difficulty,
        "duration_minutes": duration
    }

    print(f"✅ Session started: {session_id}")
    
    return {
        "session_id": session_id,
        "first_question": {
            "text_response": result.get("text_response", ""),
            "voice_response": result.get("voice_response", result.get("text_response", "")),
            "end": result.get("end", False)
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
    
    # Verify ownership
    if session.get("user_id") != request.user_id:
        return {"error": "Unauthorized"}, 403

    # Build conversation
    conv = session["messages"]
    
    # Update counters
    session["exchange_count"] += 1
    if not user_msg.startswith("["):
        session["question_count"] = session.get("question_count", 0) + 1

    # Add context for AI
    elapsed_min = (time.time() - session["created_at"]) / 60
    context = f"""
[INTERNAL CONTEXT - DO NOT MENTION]
- Exchange: {session['exchange_count']}
- Questions answered: {session.get('question_count', 0)}
- Time elapsed: {elapsed_min:.1f} min
[END CONTEXT]

User message: {user_msg}
"""

    conv.append({"role": "user", "content": context})

    result = call_claude(session["system_prompt"], conv)

    if "error" in result:
        return {"error": result["error"]}, 500

    # Update conversation with clean message
    conv[-1] = {"role": "user", "content": user_msg}
    conv.append({"role": "assistant", "content": result.get("text_response", "")})
    
    session["messages"] = conv

    print(f"📊 Session {session_id}: Exchange {session['exchange_count']}")
    
    # Return full response (including summary fields if present)
    response = {
        "text_response": result.get("text_response", ""),
        "voice_response": result.get("voice_response", result.get("text_response", "")),
        "end": result.get("end", False)
    }
    
    # Add summary fields if present
    if result.get("strengths"):
        response["strengths"] = result.get("strengths")
    if result.get("weaknesses"):
        response["weaknesses"] = result.get("weaknesses")
    if result.get("score"):
        response["score"] = result.get("score")
    if result.get("communication_score"):
        response["communication_score"] = result.get("communication_score")
    if result.get("technical_score"):
        response["technical_score"] = result.get("technical_score")
    if result.get("confidence_score"):
        response["confidence_score"] = result.get("confidence_score")
    if result.get("behavior_score"):
        response["behavior_score"] = result.get("behavior_score")
    if result.get("overall_impression"):
        response["overall_impression"] = result.get("overall_impression")
    if result.get("recommendations"):
        response["recommendations"] = result.get("recommendations")
    if "selected" in result:
        response["selected"] = result.get("selected")
    
    return response, 200


@app.route("/api/tts", methods=["POST", "OPTIONS"])
@verify_firebase_token
def tts():
    data = request.json or {}
    text = data.get("text", "")
    voice_style = data.get("voice_style", "male")

    if not text:
        return {"error": "No text"}, 400
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ' '.join(text.split())
    
    api_key = get_next_eleven_key()
    if not api_key:
        return {"error": "Missing ElevenLabs key"}, 500

    voice_id = VOICE_MAP.get(voice_style, VOICE_MAP["male"])

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            json=payload, 
            headers=headers,
            timeout=30
        )

        if resp.status_code != 200:
            print("❌ TTS error:", resp.status_code, resp.text)
            return {"error": "TTS failed"}, 500

        return Response(resp.content, mimetype="audio/mpeg")
    
    except Exception as e:
        print("❌ TTS exception:", str(e))
        return {"error": str(e)}, 500


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        headers['Access-Control-Allow-Headers'] = "Content-Type, Authorization"
        headers['Access-Control-Allow-Methods'] = "GET, POST, OPTIONS"
        return response


# -------------------------------------------------------
# START SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("🚀 Claude Sonnet 4 backend")
    print(f"✅ Model: {ANTHROPIC_MODEL}")
    print(f"✅ ElevenLabs: {len(ELEVEN_KEYS)} keys")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
