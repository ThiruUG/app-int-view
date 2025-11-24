# Corrected Full Claude + ElevenLabs Backend (Claude Messages API)

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
     methods=["GET", "POST", "OPTIONS"])

try:
    cred_json = os.getenv('FIREBASE_CREDENTIALS', '{}')
    cred_dict = json.loads(cred_json) if cred_json and cred_json != '{}' else None
    if cred_dict:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("Firebase init OK")
except Exception as e:
    print("Firebase init error", e)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
ANTHROPIC_VERSION = os.getenv('ANTHROPIC_VERSION', '2023-06-01')

ELEVEN_KEYS = [k.strip() for k in os.getenv('ELEVEN_KEYS', '').split(',') if k.strip()]
VOICE_MAP = {
    "male": os.getenv("ELEVEN_VOICE_MALE", "pNInz6obpgDQGcFmaJgB"),
    "female": os.getenv("ELEVEN_VOICE_FEMALE", "21m00Tcm4TlvDq8ikWAM")
}
key_indices = {'eleven': 0}
sessions = {}
user_sessions = {}


def verify_firebase_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return f(*args, **kwargs)
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = header.split(" ")[1]
        try:
            info = auth.verify_id_token(token)
            request.user_id = info['uid']
            return f(*args, **kwargs)
        except:
            return jsonify({"error": "Invalid token"}), 401
    return wrapper


def get_next_eleven_key():
    if not ELEVEN_KEYS:
        return None
    idx = key_indices['eleven']
    key_indices['eleven'] = (idx + 1) % len(ELEVEN_KEYS)
    return ELEVEN_KEYS[idx]


# Claude Messages API -------------------
def call_claude(prompt, max_tokens=500):
    if not ANTHROPIC_API_KEY:
        return {"error": "No Anthropic API key"}

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json"
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            return {"error": f"Claude API error {r.status_code}: {r.text}"}
        data = r.json()
        return {"text": data["content"][0]["text"]}
    except Exception as e:
        return {"error": str(e)}


# Prompt -------------------
def create_system_prompt(domain, role, interview_type, difficulty):
    return f"You are an interview AI for domain {domain}, role {role}, type {interview_type}, difficulty {difficulty}. Ask one question at a time and return JSON responses."


@app.route('/api/start-session', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def start_session():
    if request.method == 'OPTIONS':
        return '', 204
    d = request.get_json() or {}
    if not d.get('domain'):
        return jsonify({"error": "missing domain"}), 400
    sid = str(uuid.uuid4())
    system_prompt = create_system_prompt(d['domain'], d['role'], d.get('interview_type'), d.get('difficulty'))

    prompt = system_prompt + "\n\nStart with warm small talk."
    res = call_claude(prompt)
    if 'error' in res:
        return jsonify({'error': res['error']}), 500

    sessions[sid] = {
        'user_id': request.user_id,
        'system_prompt': system_prompt,
        'messages': []
    }

    return jsonify({
        "session_id": sid,
        "first_question": {
            "text_response": res['text'],
            "voice_response": res['text']
        }
    }), 200


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
@verify_firebase_token
def chat():
    if request.method == 'OPTIONS': return '', 204
    d = request.get_json() or {}
    sid = d.get('session_id')
    umsg = d.get('user_message')
    s = sessions.get(sid)
    if not s: return jsonify({"error": "session missing"}), 404

    convo = s['system_prompt'] + "\n\n" + "Human: " + umsg + "\nAssistant:"
    res = call_claude(convo)
    if 'error' in res: return jsonify({'error': res['error']}), 500

    return jsonify({
        "text_response": res['text'],
        "voice_response": res['text'],
        "end": False
    })


@app.route('/api/tts', methods=['POST'])
@verify_firebase_token
def tts():
    d = request.get_json() or {}
    txt = d.get('text')
    key = get_next_eleven_key()
    if not key: return jsonify({"error": "No TTS key"}), 500
    vid = VOICE_MAP['male']
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                      headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                      json={"text": txt}, timeout=40)
    if r.status_code != 200:
        return jsonify({"error": r.text}), r.status_code
    return Response(r.content, mimetype='audio/mpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
