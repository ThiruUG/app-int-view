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

# CORS - allow your frontend origins
CORS(app,
     resources={r"/*": {
         "origins": [
             "https://eightfoldai-chat.netlify.app",
             "http://localhost:3000",
             "http://127.0.0.1:3000",
             "http://localhost:5500",
             "http://127.0.0.1:5500"
         ]
     }},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"]
)

# Initialize Firebase Admin SDK if credentials provided
try:
    cred_dict = json.loads(os.getenv('FIREBASE_CREDENTIALS', '{}'))
    if cred_dict:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK initialized")
    else:
        print("⚠️  Firebase not initialized - set FIREBASE_CREDENTIALS env variable")
except Exception as e:
    print(f"⚠️  Firebase initialization error: {str(e)}")

# TTS and LLM configuration
ELEVEN_KEYS = [k.strip() for k in os.getenv('ELEVEN_KEYS', '').split(',') if k.strip()]
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-7-sonnet')

VOICE_MAP = {
    "male": os.getenv("ELEVEN_VOICE_MALE", "pNInz6obpgDQGcFmaJgB"),
    "female": os.getenv("ELEVEN_VOICE_FEMALE", "21m00Tcm4TlvDq8ikWAM"),
}

# In-memory session storage (for demo). Replace with DB for production.
sessions = {}
user_sessions = {}

# ----------------------
# Utility / Auth helpers
# ----------------------

def verify_firebase_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow OPTIONS preflight
        if request.method == "OPTIONS":
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized - No token provided"}), 401

        token = auth_header.split("Bearer ")[1]
        try:
            decoded = auth.verify_id_token(token)
            request.user_id = decoded["uid"]
            request.user_email = decoded.get("email", "unknown")
            return f(*args, **kwargs)
        except Exception as e:
            print(f"❌ Token verification failed: {str(e)}")
            return jsonify({"error": "Unauthorized - Invalid token"}), 401

    return decorated_function

# ----------------------
# Content safety helper
# ----------------------

def detect_inappropriate_content(text):
    text_lower = (text or "").lower()
    profanity_patterns = ['fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'stupid ai', 'dumb ai', 'idiot']
    spam_patterns = ['spam', 'buy now', 'click here', 'win prize']
    harassment_patterns = ['hate you', 'kill', 'threat', 'attack']

    is_profanity = any(word in text_lower for word in profanity_patterns)
    is_spam = any(word in text_lower for word in spam_patterns)
    is_harassment = any(word in text_lower for word in harassment_patterns)
    has_gibberish = bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text_lower))
    is_too_short = len((text or "").strip()) <= 2 and not (text or "").strip().isalpha()

    return {
        'is_inappropriate': is_profanity or is_harassment,
        'is_spam': is_spam,
        'is_too_short': is_too_short,
        'needs_redirection': is_profanity or is_spam or is_harassment
    }

# ----------------------
# Claude (Anthropic) LLM call
# ----------------------

def call_llm(messages, system_prompt, max_retries=3):
    """Call Anthropic Claude directly using REST API.
    messages: list of dicts with keys {'role': 'user'|'assistant'|'system', 'content': '...'}
    system_prompt: string
    Returns parsed JSON-like dict expected by frontend.
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "Missing ANTHROPIC_API_KEY env variable"}

    # Build Anthropic "messages" payload: prepend system
    anth_messages = []
    anth_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        # ensure role is one of user/assistant/system
        role = m.get('role', 'user')
        content = m.get('content', '')
        anth_messages.append({"role": role, "content": content})

    url = "https://api.anthropic.com/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {ANTHROPIC_API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': ANTHROPIC_MODEL,
        'messages': anth_messages,
        'temperature': 0.7,
        'max_tokens_to_sample': 500
    }

    for attempt in range(max_retries):
        try:
            print(f"🔄 Claude attempt {attempt+1}/{max_retries}")
            resp = requests.post(url, json=payload, headers=headers, timeout=30)

            if resp.status_code != 200:
                text = resp.text
                print(f"❌ Claude HTTP {resp.status_code}: {text}")
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries-1:
                    time.sleep(2)
                    continue
                return {"error": f"Claude API error {resp.status_code}: {text}"}

            data = resp.json()

            # Try common locations for text output (supporting multiple Anthropic response shapes)
            raw_text = None
            if isinstance(data, dict):
                # new-style: data['completion'] or data['choices'][0]['message']['content']
                if 'completion' in data and isinstance(data['completion'], str):
                    raw_text = data['completion']
                elif 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                    ch = data['choices'][0]
                    if 'message' in ch and 'content' in ch['message']:
                        raw_text = ch['message']['content']
                    elif 'text' in ch:
                        raw_text = ch['text']
                elif 'output' in data and isinstance(data['output'], str):
                    raw_text = data['output']

            if raw_text is None:
                # fallback: treat whole response as text
                raw_text = resp.text

            # Attempt to extract JSON block from the assistant response
            try:
                # Sometimes model returns a JSON block inside markdown fences - extract if present
                if '```json' in raw_text:
                    raw_text = raw_text.split('```json',1)[1].split('```',1)[0].strip()

                parsed = json.loads(raw_text)
                # Ensure required keys
                if 'text_response' not in parsed:
                    parsed['text_response'] = raw_text
                if 'voice_response' not in parsed:
                    parsed['voice_response'] = parsed['text_response']
                if 'end' not in parsed:
                    parsed['end'] = False
                return parsed
            except Exception:
                # Try to find a JSON object inside text using regex
                json_match = re.search(r'\{[\s\S]*\}', raw_text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        if 'text_response' not in parsed:
                            parsed['text_response'] = raw_text
                        if 'voice_response' not in parsed:
                            parsed['voice_response'] = parsed['text_response']
                        if 'end' not in parsed:
                            parsed['end'] = False
                        return parsed
                    except Exception:
                        pass

                # Fallback: return raw_text as text_response
                return {
                    'text_response': raw_text,
                    'voice_response': raw_text,
                    'end': False
                }

        except Exception as e:
            print(f"❌ Claude request exception: {e}")
            if attempt < max_retries-1:
                time.sleep(2)
                continue
            return {"error": f"Claude request failed: {str(e)}"}

    return {"error": "Max retries exceeded for Claude API"}

# ----------------------
# System prompt creator
# ----------------------

def create_system_prompt(domain, role, interview_type, difficulty):
    return f"""You are \"AI Interview Practitioner, Claude edition\".\n
The user has selected:\n- Domain: {domain}\n- Role: {role}\n- Interview Type: {interview_type}\n- Difficulty: {difficulty}\n
Follow the interview flow and ALWAYS return EXACT valid JSON as specified to the frontend.\n"""

# ----------------------
# API endpoints
# ----------------------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'online',
        'service': 'EightFold.ai Interview Backend (Claude)',
        'version': '3.0',
        'endpoints': {
            'health': '/health',
            'start_session': '/api/start-session',
            'chat': '/api/chat',
            'tts': '/api/tts'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'firebase_initialized': len(firebase_admin._apps) > 0,
        'tts': f'ElevenLabs ({len(ELEVEN_KEYS)} keys configured)',
        'llm': f'Anthropic Claude ({ANTHROPIC_MODEL})',
        'active_sessions': len(sessions),
        'cors_enabled': True
    })

@app.route('/api/start-session', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def start_session():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json or {}
    domain = data.get('domain')
    role = data.get('role')
    interview_type = data.get('interview_type', 'Mixed')
    difficulty = data.get('difficulty', 'Intermediate')
    duration = data.get('duration', 15)

    if not domain or not role:
        return jsonify({'error': 'Domain and role are required'}), 400

    session_id = str(uuid.uuid4())
    system_prompt = create_system_prompt(domain, role, interview_type, difficulty)

    messages = [{'role': 'user', 'content': 'Start the interview with warm small talk as instructed.'}]

    ai_response = call_llm(messages, system_prompt)
    if 'error' in ai_response:
        return jsonify({'error': ai_response['error']}), 500

    # store session
    sessions[session_id] = {
        'user_id': request.user_id,
        'user_email': request.user_email,
        'domain': domain,
        'role': role,
        'interview_type': interview_type,
        'difficulty': difficulty,
        'duration_minutes': duration,
        'system_prompt': system_prompt,
        'messages': messages + [{'role': 'assistant', 'content': json.dumps(ai_response)}],
        'created_at': time.time(),
        'exchange_count': 0,
        'question_count': 0,
        'inappropriate_count': 0,
        'redirect_count': 0
    }

    if request.user_id not in user_sessions:
        user_sessions[request.user_id] = []
    user_sessions[request.user_id].append(session_id)

    return jsonify({'session_id': session_id, 'first_question': ai_response}), 200

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def chat():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json or {}
    session_id = data.get('session_id')
    user_message = data.get('user_message')

    if not session_id or user_message is None:
        return jsonify({'error': 'Session ID and user message are required'}), 400

    if session_id not in sessions:
        return jsonify({'error': 'Session not found or expired'}), 404

    session = sessions[session_id]
    if session['user_id'] != request.user_id:
        return jsonify({'error': 'Unauthorized - Session does not belong to user'}), 403

    # safety checks
    if not user_message.startswith('['):
        checks = detect_inappropriate_content(user_message)
        if checks['needs_redirection']:
            session['inappropriate_count'] += 1
            session['redirect_count'] += 1
            if session['inappropriate_count'] >= 3:
                user_message = '[END_INTERVIEW_INAPPROPRIATE_BEHAVIOR]'

    session['messages'].append({'role': 'user', 'content': user_message})
    session['exchange_count'] += 1

    # Add context block to help model
    elapsed_minutes = (time.time() - session['created_at']) / 60
    time_remaining_minutes = session['duration_minutes'] - elapsed_minutes

    context_info = f"""
[CONTEXT - DO NOT MENTION TO USER]
- Total exchanges: {session['exchange_count']}
- Interview questions answered: {session.get('question_count', 0)}
- Inappropriate behavior count: {session.get('inappropriate_count', 0)}
- Redirect count: {session.get('redirect_count', 0)}
- Duration: {session['duration_minutes']} min
- Time elapsed: {elapsed_minutes:.1f} min
- Time remaining: {time_remaining_minutes:.1f} min
[END CONTEXT]

User message: {user_message}
"""

    # Replace last added user message with context info for model
    session['messages'][-1] = {'role': 'user', 'content': context_info}

    ai_response = call_llm(session['messages'], session['system_prompt'])
    if 'error' in ai_response:
        return jsonify({'error': ai_response['error']}), 500

    # restore actual user message in session history and append assistant response
    session['messages'][-1] = {'role': 'user', 'content': user_message}
    session['messages'].append({'role': 'assistant', 'content': json.dumps(ai_response)})

    if ai_response.get('end', False):
        session['ended_at'] = time.time()

    return jsonify(ai_response), 200

@app.route('/api/tts', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def tts():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json() or {}
    text = data.get('text')
    voice_style = (data.get('voice_style', 'male') or 'male').lower()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    api_key = ELEVEN_KEYS[0] if ELEVEN_KEYS else None
    if not api_key:
        return jsonify({'error': 'No ElevenLabs API key configured'}), 500

    voice_id = VOICE_MAP.get(voice_style, VOICE_MAP['male'])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        'xi-api-key': api_key,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg'
    }

    payload = {
        'text': text,
        'model_id': 'eleven_turbo_v2',
        'voice_settings': {'stability': 0.35, 'similarity_boost': 0.7}
    }

    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        print('❌ ElevenLabs Error:', resp.text)
        return jsonify({'error': 'TTS failed', 'details': resp.text}), resp.status_code

    return Response(resp.content, mimetype='audio/mpeg', headers={
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    })

# ----------------------
# Run server
# ----------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting EightFold.ai Interview Backend (Claude) - Model: {ANTHROPIC_MODEL}")
    print(f"✅ Firebase Auth: {'Enabled' if len(firebase_admin._apps) > 0 else 'Disabled'}")
    print(f"✅ TTS: ElevenLabs ({len(ELEVEN_KEYS)} keys)")
    print(f"✅ LLM: Anthropic Claude ({ANTHROPIC_MODEL})")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
