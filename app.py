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

# ----------------------
# Firebase initialization
# ----------------------
try:
    cred_json = os.getenv('FIREBASE_CREDENTIALS', '{}')
    cred_dict = json.loads(cred_json) if cred_json and cred_json != '{}' else None
    if cred_dict:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK initialized")
    else:
        print("⚠️  Firebase not initialized - set FIREBASE_CREDENTIALS env variable")
except Exception as e:
    print(f"⚠️  Firebase initialization error: {str(e)}")

# ----------------------
# API keys and config
# ----------------------
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
ANTHROPIC_VERSION = os.getenv('ANTHROPIC_VERSION', '2025-11-01')

ELEVEN_KEYS = [k.strip() for k in os.getenv('ELEVEN_KEYS', '').split(',') if k.strip()]
VOICE_MAP = {
    "male": os.getenv("ELEVEN_VOICE_MALE", "pNInz6obpgDQGcFmaJgB"),
    "female": os.getenv("ELEVEN_VOICE_FEMALE", "21m00Tcm4TlvDq8ikWAM"),
}

# rotation indices
key_indices = {'eleven': 0}

# session storage
sessions = {}
user_sessions = {}

# ----------------------
# Helpers
# ----------------------

def verify_firebase_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized - No token provided'}), 401

        token = auth_header.split('Bearer ')[1]
        try:
            decoded = auth.verify_id_token(token)
            request.user_id = decoded['uid']
            request.user_email = decoded.get('email', 'unknown')
            return f(*args, **kwargs)
        except Exception as e:
            print(f"❌ Token verification failed: {str(e)}")
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401

    return decorated_function


def get_next_eleven_key():
    if not ELEVEN_KEYS:
        return None
    idx = key_indices['eleven']
    k = ELEVEN_KEYS[idx]
    key_indices['eleven'] = (idx + 1) % len(ELEVEN_KEYS)
    return k


def detect_inappropriate_content(text):
    txt = (text or '').lower()
    profanity = ['fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'idiot', 'stupid']
    spam = ['buy now', 'click here', 'win prize']
    is_prof = any(w in txt for w in profanity)
    is_spam = any(w in txt for w in spam)
    is_too_short = len((text or '').strip()) <= 2
    return {'is_inappropriate': is_prof, 'is_spam': is_spam, 'is_too_short': is_too_short, 'needs_redirection': is_prof or is_spam}

# ----------------------
# Anthropic (Claude) wrapper
# ----------------------

def call_claude(prompt, max_retries=3, timeout=30, temperature=0.7, max_tokens=8000):
    """
    Simple wrapper to call Anthropic's completion endpoint for Claude.
    The prompt should include system/instructions + conversation formatted.
    """
    if not ANTHROPIC_API_KEY:
        return {'error': 'Anthropic API key not configured'}

    url = 'https://api.anthropic.com/v1/complete'

    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': ANTHROPIC_VERSION,
        'Content-Type': 'application/json'
    }

    body = {
        'model': ANTHROPIC_MODEL,
        'prompt': prompt,
        'max_tokens_to_sample': 512,
        'temperature': float(temperature),
        'top_k': None
    }

    for attempt in range(max_retries):
        try:
            print(f"🔄 Claude attempt {attempt+1}/{max_retries} model={ANTHROPIC_MODEL}")
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # Anthropic returns 'completion' key
                text = data.get('completion') or data.get('response') or ''
                return {'text': text, 'raw': data}
            else:
                print(f"❌ Claude HTTP {resp.status_code}: {resp.text}")
                # retry on 5xx
                if resp.status_code >= 500 and attempt < max_retries-1:
                    time.sleep(1 + attempt)
                    continue
                return {'error': f"Claude API error {resp.status_code}: {resp.text}"}
        except Exception as e:
            print(f"❌ Claude Exception: {e}")
            if attempt < max_retries-1:
                time.sleep(1 + attempt)
                continue
            return {'error': str(e)}

    return {'error': 'Max retries exceeded'}

# ----------------------
# System prompt builder (keeps your previous elaborate prompt)
# ----------------------

def create_system_prompt(domain, role, interview_type, difficulty):
    # Use the same detailed prompt structure you had, condensed for Claude
    return f"""
You are a professional mock interview coach (text + voice) for domain={domain}, role={role}, type={interview_type}, difficulty={difficulty}.
Follow structured behavior: warm small talk (2 exchanges), then ask one question at a time, be encouraging, handle edge-cases (inappropriate, gibberish, short answers), and produce final summary after 5-7 questions.
Respond in JSON format as specified by the app (text_response, voice_response, end) when replying.
"""

# ----------------------
# API endpoints
# ----------------------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'online',
        'service': 'EightFold.ai Interview Backend (Claude)',
        'version': '3.0'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'anthropic_configured': bool(ANTHROPIC_API_KEY),
        'eleven_keys': len(ELEVEN_KEYS)
    }), 200

@app.route('/api/start-session', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def start_session():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json() or {}
    domain = data.get('domain')
    role = data.get('role')
    interview_type = data.get('interview_type', 'Mixed')
    difficulty = data.get('difficulty', 'Intermediate')
    duration = int(data.get('duration', 15))

    if not domain or not role:
        return jsonify({'error': 'domain and role required'}), 400

    session_id = str(uuid.uuid4())
    system_prompt = create_system_prompt(domain, role, interview_type, difficulty)

    # First prompt: instruct Claude to produce the JSON output the frontend expects
    prompt = system_prompt + '

Human: Start the interview with warm small talk as instructed.
Assistant:'

    result = call_claude(prompt)
    if 'error' in result:
        return jsonify({'error': result['error']}), 500

    # Store session
    sessions[session_id] = {
        'user_id': request.user_id,
        'user_email': request.user_email,
        'domain': domain,
        'role': role,
        'interview_type': interview_type,
        'difficulty': difficulty,
        'duration_minutes': duration,
        'system_prompt': system_prompt,
        'messages': [{'role': 'assistant', 'content': result.get('text', '')}],
        'created_at': time.time(),
        'exchange_count': 0,
        'question_count': 0,
        'inappropriate_count': 0
    }

    if request.user_id not in user_sessions:
        user_sessions[request.user_id] = []
    user_sessions[request.user_id].append(session_id)

    print(f"✅ Session started: {session_id} for user {request.user_email}")

    # The frontend expects first_question with text_response and voice_response
    first_question = {
        'text_response': result.get('text', ''),
        'voice_response': result.get('text', '')
    }

    return jsonify({'session_id': session_id, 'first_question': first_question}), 200

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def chat():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json() or {}
    session_id = data.get('session_id')
    user_message = data.get('user_message')
    voice_style = data.get('voice_style', 'male')

    if not session_id or not user_message:
        return jsonify({'error': 'session_id and user_message required'}), 400

    if session_id not in sessions:
        return jsonify({'error': 'session not found'}), 404

    session = sessions[session_id]
    if session['user_id'] != request.user_id:
        return jsonify({'error': 'unauthorized - session mismatch'}), 403

    # Basic content checks
    if not user_message.startswith('['):
        check = detect_inappropriate_content(user_message)
        if check['needs_redirection']:
            session['inappropriate_count'] += 1
            if session['inappropriate_count'] >= 3:
                # force end
                user_message = '[END_INTERVIEW_INAPPROPRIATE_BEHAVIOR]'

    # Append and build prompt context (keep it short to avoid token explosion)
    session['messages'].append({'role': 'user', 'content': user_message})
    session['exchange_count'] += 1

    # Build a compact conversation string
    convo = ''
    # start with system prompt
    convo += session['system_prompt'] + '

'
    # include last few messages (e.g., last 8)
    recent = session['messages'][-8:]
    for m in recent:
        role = 'Human' if m['role'] == 'user' else 'Assistant'
        convo += f"{role}: {m['content']}
"
    convo += 'Assistant:'

    result = call_claude(convo)
    if 'error' in result:
        return jsonify({'error': result['error']}), 500

    text = result.get('text', '')

    # store assistant reply
    session['messages'].append({'role': 'assistant', 'content': text})

    # If assistant indicates end (we rely on frontend special commands or content parsing)
    end_flag = False
    if '[END' in text or 'Thank you for completing this interview' in text:
        end_flag = True
        session['ended_at'] = time.time()

    # prepare response JSON compatible with frontend
    response_json = {
        'text_response': text,
        'voice_response': text,
        'end': end_flag
    }

    # If end, optionally compute summary elements (not implemented: LLM can be asked to create full summary)
    return jsonify(response_json), 200

@app.route('/api/tts', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def tts():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json() or {}
    text = data.get('text')
    voice_style = data.get('voice_style', 'male').lower()

    if not text:
        return jsonify({'error': 'no text provided'}), 400

    api_key = get_next_eleven_key() or (ELEVEN_KEYS[0] if ELEVEN_KEYS else None)
    if not api_key:
        return jsonify({'error': 'no ElevenLabs API key configured'}), 500

    voice_id = VOICE_MAP.get(voice_style, VOICE_MAP['male'])
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'

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

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        print('❌ ElevenLabs Error:', resp.status_code, resp.text)
        return jsonify({'error': 'TTS failed', 'details': resp.text}), resp.status_code

    return Response(resp.content, mimetype='audio/mpeg', headers={
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    })

@app.route('/api/user-sessions', methods=['GET', 'OPTIONS'])
@verify_firebase_token
def get_user_sessions():
    if request.method == 'OPTIONS':
        return '', 204
    user_id = request.user_id
    ids = user_sessions.get(user_id, [])
    out = []
    for sid in ids:
        s = sessions.get(sid)
        if s:
            out.append({
                'session_id': sid,
                'domain': s['domain'],
                'role': s['role'],
                'interview_type': s['interview_type'],
                'difficulty': s['difficulty'],
                'created_at': s['created_at'],
                'exchange_count': s['exchange_count'],
                'ended': 'ended_at' in s
            })
    return jsonify({'user_id': user_id, 'sessions': out}), 200

# Cleanup

def cleanup_old_sessions():
    now = time.time()
    to_delete = []
    for sid, s in list(sessions.items()):
        if now - s['created_at'] > 86400:
            to_delete.append(sid)
        elif 'ended_at' in s and now - s['ended_at'] > 3600:
            to_delete.append(sid)
    for sid in to_delete:
        uid = sessions[sid]['user_id']
        if uid in user_sessions:
            user_sessions[uid] = [x for x in user_sessions[uid] if x != sid]
        del sessions[sid]
    if to_delete:
        print(f"🧹 Cleaned up {len(to_delete)} sessions")

# Error handlers

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not found'}), 404

@app.errorhandler(500)
def internal_err(e):
    return jsonify({'error': 'internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print('🚀 EightFold.ai Claude backend starting')
    print(f'🔎 Anthropic configured: {bool(ANTHROPIC_API_KEY)} model={ANTHROPIC_MODEL}')
    print(f'🔊 ElevenLabs keys: {len(ELEVEN_KEYS)}')
    app.run(host='0.0.0.0', port=port, debug=False)
