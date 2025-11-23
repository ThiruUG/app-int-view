from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import requests
import os
import uuid
from io import BytesIO
import time
import re
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
# OPTION 1: Using service account JSON file
# cred = credentials.Certificate('path/to/serviceAccountKey.json')
# firebase_admin.initialize_app(cred)

# OPTION 2: Using environment variable (recommended for production)
cred_dict = json.loads(os.getenv('FIREBASE_CREDENTIALS', '{}'))
if cred_dict:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
else:
    print("⚠️  Firebase not initialized - set FIREBASE_CREDENTIALS env variable")

# API Keys configuration
ELEVEN_KEYS = [k.strip() for k in os.getenv('ELEVEN_KEYS', '').split(',') if k.strip()]
OPENROUTER_KEYS = [k.strip() for k in os.getenv('OPENROUTER_KEYS', '').split(',') if k.strip()]
OPENAI_KEYS = [k.strip() for k in os.getenv('OPENAI_KEYS', '').split(',') if k.strip()]

# Key rotation indices
key_indices = {'eleven': 0, 'openrouter': 0, 'openai': 0}

# Session storage
sessions = {}

def verify_firebase_token(f):
    """Decorator to verify Firebase authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Unauthorized - No token provided"}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        try:
            decoded_token = auth.verify_id_token(token)
            request.user_id = decoded_token['uid']
            request.user_email = decoded_token.get('email', 'unknown')
            return f(*args, **kwargs)
        except Exception as e:
            print(f"❌ Token verification failed: {str(e)}")
            return jsonify({"error": "Unauthorized - Invalid token"}), 401
    
    return decorated_function

def get_next_key(service):
    """Get next API key with rotation"""
    if service == 'eleven' and ELEVEN_KEYS:
        idx = key_indices['eleven']
        key = ELEVEN_KEYS[idx]
        key_indices['eleven'] = (idx + 1) % len(ELEVEN_KEYS)
        return key
    elif service == 'openrouter' and OPENROUTER_KEYS:
        idx = key_indices['openrouter']
        key = OPENROUTER_KEYS[idx]
        key_indices['openrouter'] = (idx + 1) % len(OPENROUTER_KEYS)
        return key
    elif service == 'openai' and OPENAI_KEYS:
        idx = key_indices['openai']
        key = OPENAI_KEYS[idx]
        key_indices['openai'] = (idx + 1) % len(OPENAI_KEYS)
        return key
    return None

def detect_inappropriate_content(text):
    """Detect inappropriate, rude, or offensive content"""
    # Convert to lowercase for checking
    text_lower = text.lower()
    
    # Patterns to detect
    profanity_patterns = ['fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'stupid ai', 'dumb ai', 'idiot']
    spam_patterns = ['spam', 'buy now', 'click here', 'win prize']
    harassment_patterns = ['hate you', 'kill', 'threat', 'attack']
    
    # Check for patterns
    is_profanity = any(word in text_lower for word in profanity_patterns)
    is_spam = any(word in text_lower for word in spam_patterns)
    is_harassment = any(word in text_lower for word in harassment_patterns)
    
    # Check for gibberish (too many consonants in a row, random characters)
    has_gibberish = bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text_lower))
    
    # Check for emoji-only or very short responses
    is_too_short = len(text.strip()) <= 2 and not text.strip().isalpha()
    
    return {
        'is_inappropriate': is_profanity or is_harassment,
        'is_spam': is_spam,
        'is_gibberish': has_gibberish,
        'is_too_short': is_too_short,
        'needs_redirection': is_profanity or is_spam or is_gibberish or is_harassment
    }

def call_llm(messages, system_prompt, max_retries=3):
    """Call LLM with retry logic and key rotation"""
    for attempt in range(max_retries):
        api_key = get_next_key('openrouter')
        if not api_key:
            return {"error": "No OpenRouter API key available"}
        
        try:
            print(f"🔄 LLM attempt {attempt + 1}/{max_retries}")
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://ai-interview-practitioner.com',
                    'X-Title': 'AI Interview Practitioner'
                },
                json={
                    'model': 'anthropic/claude-3.5-haiku',
                    'messages': [
                        {'role': 'system', 'content': system_prompt}
                    ] + messages,
                    'temperature': 0.7,
                    'max_tokens': 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                try:
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    parsed = json.loads(content)
                    
                    if 'text_response' not in parsed:
                        parsed['text_response'] = content
                    if 'voice_response' not in parsed:
                        parsed['voice_response'] = parsed['text_response']
                    if 'end' not in parsed:
                        parsed['end'] = False
                    
                    print(f"✅ LLM Success")
                    return parsed
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        print(f"⚠️ JSON parse failed, retrying...")
                        continue
                    return {
                        "text_response": content,
                        "voice_response": content,
                        "end": False
                    }
            
            elif response.status_code in [429, 500, 502, 503, 504]:
                print(f"⚠️ LLM Status {response.status_code}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            return {"error": f"LLM API error: {response.status_code}"}
        
        except Exception as e:
            print(f"❌ LLM Exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"error": f"LLM request failed: {str(e)}"}
    
    return {"error": "Max retries exceeded"}

def create_system_prompt(domain, role, interview_type, difficulty):
    """Create system prompt with edge case handling"""
    return f"""You are "AI Interview Practitioner," a professional mock interview coach with dual communication modes: text and voice.

The user has selected:
- Domain: {domain}
- Role: {role}
- Interview Type: {interview_type}
- Difficulty: {difficulty}

OPENING BEHAVIOR (FIRST 2-3 EXCHANGES):
1. FIRST MESSAGE: Start with warm small talk (1-2 sentences). Ask how they're feeling about the interview today.
2. SECOND MESSAGE: After their response, acknowledge it warmly and ask one more casual question.
3. THIRD MESSAGE: After their second response, smoothly transition: "Great! Now let's dive into the interview. Here's my first question..." and then ask your FIRST actual interview question

DURING INTERVIEW (AFTER SMALL TALK):
- Ask ONLY ONE question at a time
- If candidate struggles, encourage: "Take your time, there's no rush"
- If they say "I don't know", respond: "That's okay, let's move to the next question"
- Be warm, encouraging, and professional
- After 5-7 meaningful interview questions (not counting small talk), provide the FINAL SUMMARY

EDGE CASE HANDLING (CRITICAL - ALWAYS MAINTAIN PROFESSIONALISM):

1. IRRELEVANT RESPONSES:
   - If user talks about unrelated topics (e.g., "I like pizza", "What's the weather?", random stories)
   - Respond: "I appreciate you sharing, but let's stay focused on the interview. Let me ask you: [repeat or rephrase current question]"
   - Never mock or be dismissive
   - Gently redirect 2-3 times, then note in evaluation: "Candidate struggled with focus"

2. RUDE/OFFENSIVE BEHAVIOR:
   - If user is rude, uses profanity, or is disrespectful
   - Respond with EXTREME professionalism: "I understand interviews can be stressful. Let's maintain a professional environment. May I continue with the next question?"
   - Never escalate or respond with rudeness
   - If behavior continues, respond: "I appreciate your time, but I think we should conclude here. Let me provide your feedback."
   - Then END interview with summary noting: "Professional conduct needs improvement"

3. GIBBERISH/NONSENSICAL INPUT:
   - If response is random characters, emojis only, or nonsensical
   - Respond: "I didn't quite catch that. Could you please rephrase your answer?"
   - After 2-3 attempts, respond: "Let's move to the next question to make the best use of our time."

4. ONE-WORD/VERY SHORT ANSWERS:
   - If user consistently gives "yes", "no", "idk", "ok" type responses
   - Respond: "Could you elaborate a bit more? I'd like to understand your thinking process."
   - Note in evaluation: "Communication skills need development - responses lacked depth"

5. ASKING AI TO DO UNETHICAL THINGS:
   - If user asks you to give them answers, cheat, or break interview rules
   - Respond: "I'm here to help you practice and improve, not to provide answers. Let's continue with the interview professionally."
   - Maintain firm but polite boundaries

6. OFF-TOPIC QUESTIONS TO AI:
   - If user asks "What's your name?", "Are you real?", "Can you do X for me?"
   - Respond: "I'm your interview coach for this session. Let's focus on helping you prepare for {role} interviews. Now, back to my question: [current question]"

7. EMOTIONAL DISTRESS:
   - If user seems very anxious, says "I can't do this", "I'm terrible", etc.
   - Respond with empathy: "Take a deep breath. This is just practice - there's no pressure here. Would you like to take a moment, or should I ask a different question?"
   - Be supportive while maintaining interview structure

8. TECHNICAL ISSUES REPORTED:
   - If user says "I can't hear", "mic not working", etc.
   - Respond: "No problem! You can type your responses. Let's continue: [current question]"

SPECIAL COMMANDS:
- If user sends "[END_INTERVIEW_TIME_UP]" or "[END_INTERVIEW_MANUAL]", immediately generate the final summary
- If user sends "[INACTIVITY_CHECK]", respond: "Are you still there? Take your time to think about the question. Would you like me to rephrase it or move to a different question?"
- If user sends "[INAPPROPRIATE_CONTENT_DETECTED]", respond: "I understand interviews can be stressful. Let's maintain a professional environment. Would you like to continue with the next question?"

EVALUATION NOTES FOR EDGE CASES:
- Track unprofessional behavior in "weaknesses" section
- Note focus issues, communication gaps, or conduct problems
- Be factual and constructive, never harsh
- If interview was disrupted significantly, mention it professionally in overall_impression

You MUST respond in EXACT JSON format:

{{
  "text_response": "<chat text - can include emojis>",
  "voice_response": "<spoken version - ABSOLUTELY NO emojis, markdown, symbols - plain English ONLY>",
  "end": false
}}

FINAL SUMMARY FORMAT (after 5-7 questions OR when ending):
{{
  "text_response": "Thank you for completing this interview! Here's your comprehensive performance summary.",
  "voice_response": "Thank you for completing this interview! Here's your comprehensive performance summary.",
  "strengths": "<3-4 specific strengths with examples>",
  "weaknesses": "<2-3 areas needing improvement>",
  "score": <0-100>,
  "communication_score": <0-100>,
  "technical_score": <0-100>,
  "confidence_score": <0-100>,
  "overall_impression": "<2-3 sentences>",
  "recommendations": "<3-4 actionable steps>",
  "selected": <true or false - based on overall performance>,
  "end": true
}}

SELECTION CRITERIA:
- Selected (true): Score >= 65, answered most questions reasonably, showed effort
- Not Selected (false): Score < 65, consistently struggled, poor communication, inappropriate behavior
- Consider difficulty level, number of questions answered, and overall engagement
"""

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "firebase_initialized": firebase_admin._apps != {},
        "stt": "Browser-based",
        "eleven_keys": len(ELEVEN_KEYS),
        "openrouter_keys": len(OPENROUTER_KEYS)
    }), 200

@app.route('/api/start-session', methods=['POST'])
@verify_firebase_token
def start_session():
    """Initialize interview session with authentication"""
    try:
        data = request.json
        domain = data.get('domain')
        role = data.get('role')
        interview_type = data.get('interview_type', 'Mixed')
        difficulty = data.get('difficulty', 'Intermediate')
        duration = data.get('duration', 15)
        
        if not domain or not role:
            return jsonify({"error": "Domain and role are required"}), 400
        
        session_id = str(uuid.uuid4())
        system_prompt = create_system_prompt(domain, role, interview_type, difficulty)
        
        messages = [{"role": "user", "content": "Start the interview with warm small talk as instructed."}]
        
        ai_response = call_llm(messages, system_prompt)
        
        if "error" in ai_response:
            return jsonify({"error": ai_response["error"]}), 500
        
        sessions[session_id] = {
            "user_id": request.user_id,
            "user_email": request.user_email,
            "domain": domain,
            "role": role,
            "interview_type": interview_type,
            "difficulty": difficulty,
            "duration_minutes": duration,
            "system_prompt": system_prompt,
            "messages": messages + [{"role": "assistant", "content": json.dumps(ai_response)}],
            "created_at": time.time(),
            "exchange_count": 0,
            "question_count": 0,
            "inappropriate_count": 0,
            "redirect_count": 0
        }
        
        print(f"✅ Session started: {session_id} for user {request.user_email}")
        return jsonify({
            "session_id": session_id,
            "first_question": ai_response
        }), 200
    
    except Exception as e:
        print(f"❌ Start session error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@verify_firebase_token
def chat():
    """Handle chat messages with edge case detection"""
    try:
        data = request.json
        session_id = data.get('session_id')
        user_message = data.get('user_message')
        voice_style = data.get('voice_style', 'male')
        
        if not session_id or not user_message:
            return jsonify({"error": "Session ID and user message are required"}), 400
        
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        session = sessions[session_id]
        
        # Verify session belongs to authenticated user
        if session['user_id'] != request.user_id:
            return jsonify({"error": "Unauthorized - Session does not belong to user"}), 403
        
        system_prompt = session['system_prompt']
        messages = session['messages']
        
        # Detect inappropriate content (skip for special commands)
        if not user_message.startswith('['):
            content_check = detect_inappropriate_content(user_message)
            
            if content_check['needs_redirection']:
                session['inappropriate_count'] += 1
                session['redirect_count'] += 1
                
                print(f"⚠️  Inappropriate content detected in session {session_id}: {content_check}")
                
                # If too many inappropriate messages, end interview
                if session['inappropriate_count'] >= 3:
                    print(f"⚠️  Ending session {session_id} due to repeated inappropriate behavior")
                    user_message = "[END_INTERVIEW_INAPPROPRIATE_BEHAVIOR]"
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        session['exchange_count'] += 1
        
        if not user_message.startswith('[') and session['exchange_count'] > 3:
            session['question_count'] = session.get('question_count', 0) + 1
        
        # Calculate time context
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
        
        messages[-1] = {"role": "user", "content": context_info}
        
        # Get AI response
        ai_response = call_llm(messages, system_prompt)
        
        if "error" in ai_response:
            return jsonify({"error": ai_response["error"]}), 500
        
        # Update messages with actual user message
        messages[-1] = {"role": "user", "content": user_message}
        messages.append({"role": "assistant", "content": json.dumps(ai_response)})
        session['messages'] = messages
        
        print(f"📊 Session {session_id}: {session['exchange_count']} exchanges, {session.get('inappropriate_count', 0)} flags")
        
        return jsonify(ai_response), 200
    
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts', methods=['POST'])
@verify_firebase_token
def text_to_speech():
    """Text-to-speech with authentication"""
    try:
        data = request.json
        text = data.get('text', '')
        voice_style = data.get('voice_style', 'male')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Clean text for TTS
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace('*', '').replace('#', '').replace('_', '').replace('`', '')
        text = ' '.join(text.split())
        
        voice_ids = {
            'male': 'pNInz6obpgDQGcFmaJgB',
            'female': '21m00Tcm4TlvDq8ikWAM'
        }
        
        voice_id = voice_ids.get(voice_style, voice_ids['male'])
        
        for attempt in range(min(3, len(ELEVEN_KEYS)) if ELEVEN_KEYS else 1):
            api_key = get_next_key('eleven')
            if not api_key:
                return jsonify({"error": "No ElevenLabs API key available"}), 500
            
            try:
                response = requests.post(
                    f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
                    headers={
                        'xi-api-key': api_key,
                        'Content-Type': 'application/json'
                    },
                    json={
                        'text': text,
                        'model_id': 'eleven_turbo_v2_5',
                        'voice_settings': {
                            'stability': 0.5,
                            'similarity_boost': 0.75
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return send_file(
                        BytesIO(response.content),
                        mimetype='audio/mpeg',
                        as_attachment=False
                    )
                elif response.status_code in [429, 500, 502, 503] and attempt < 2:
                    time.sleep(1)
                    continue
            
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
        
        return jsonify({"error": "TTS failed after retries"}), 500
    
    except Exception as e:
        print(f"❌ TTS error: {str(e)}")
        return jsonify({"error": str(e)}), 500
  if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting AI Interview Practitioner Backend")
    print(f"✅ Firebase Auth: {'Enabled' if firebase_admin._apps else 'Disabled'}")
    print(f"✅ STT: Browser-based (Web Speech API)")
    print(f"✅ TTS: ElevenLabs ({len(ELEVEN_KEYS)} keys)")
    print(f"✅ LLM: OpenRouter ({len(OPENROUTER_KEYS)} keys)")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    # For production use gunicorn instead
    app.run(host='0.0.0.0', port=port, debug=False) 
