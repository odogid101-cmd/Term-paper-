# app.py (API-only, no static serving)
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from faster_whisper import WhisperModel # NEW
import psycopg2
import psycopg2.errors
import os
import secrets
import datetime
import requests
import logging
import tempfile # NEW

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

DATABASE_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
PORT = int(os.getenv("PORT", 5000))
TEST_EMAIL_KEY = os.getenv("TEST_EMAIL_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

if not RESEND_API_KEY or not FROM_EMAIL:
    logging.warning("RESEND_API_KEY and/or FROM_EMAIL not set — send_email will fail until they are provided")

# NEW: Load Whisper model once on boot
print("Loading Whisper model... this takes 1-2 min on first boot")
try:
    model = WhisperModel("base", device="cpu", compute_type="int8") # 142MB, fits free tier
    print("Whisper model loaded!")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    model = None

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                is_verified BOOLEAN DEFAULT FALSE
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS email_verifications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                code TEXT,
                purpose TEXT,
                expires_at TIMESTAMP
            )
            """)

init_db()

def generate_code(prefix: str, digits: int = 6) -> str:
    number = secrets.randbelow(10 ** digits)
    return f"{prefix}{number:0{digits}d}"

def send_email(to_email: str, code: str) -> bool:
    if not RESEND_API_KEY or not FROM_EMAIL:
        logging.error("Email not sent: RESEND_API_KEY or FROM_EMAIL missing")
        return False
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={"from": FROM_EMAIL, "to": to_email, "subject": "Your verification code", "html": f"<h2>{code}</h2><p>This code expires in 10 minutes.</p>"},
            timeout=10
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logging.exception("Failed to send email: %s", e)
        return False

def _required_fields(data: dict, *fields):
    missing = [f for f in fields if not data.get(f)]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    return True, None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API running"}), 200

# NEW ENDPOINT: Voice to Text
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if model is None:
        return jsonify({"error": "Whisper model not loaded"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    try:
        segments, info = model.transcribe(temp_path, language="en")
        text = " ".join([segment.text for segment in segments])
        logging.info(f"Transcribed: {text}")
        return jsonify({"text": text.strip()})

    except Exception as e:
        logging.exception("Transcription failed: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "username", "email", "password")
    if not ok:
        return jsonify({"error": err}), 400
    username = data["username"].strip()
    email = data["email"].strip().lower()
    password_hash = generate_password_hash(data["password"])
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id", (username, email, password_hash))
                user_id = cur.fetchone()[0]
                code = generate_code("ver", digits=6)
                expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                cur.execute("INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (%s, %s, %s, %s)", (user_id, code, "signup", expires))
        send_email(email, code)
        return jsonify({"message": "User registered; verification code sent if email delivery succeeded."}), 201
    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "Username or email already exists"}), 409
    except Exception as e:
        logging.exception("Registration failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/verify-email", methods=["POST"])
def verify_email():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email", "code")
    if not ok: return jsonify({"error": err}), 400
    email = data["email"].strip().lower()
    code = data["code"].strip()
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "User not found"}), 404
                user_id = row[0]
                cur.execute("SELECT id, expires_at FROM email_verifications WHERE user_id=%s AND code=%s AND purpose=%s ORDER BY id DESC LIMIT 1", (user_id, code, "signup"))
                ver = cur.fetchone()
                if not ver: return jsonify({"error": "Invalid code"}), 400
                if ver[1] < datetime.datetime.utcnow(): return jsonify({"error": "Code expired"}), 400
                cur.execute("UPDATE users SET is_verified = TRUE WHERE id = %s", (user_id,))
                cur.execute("DELETE FROM email_verifications WHERE user_id=%s AND purpose=%s", (user_id, "signup"))
        return jsonify({"message": "Email verified"}), 200
    except Exception as e:
        logging.exception("Email verification failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email: return jsonify({"error": "Missing email"}), 400
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "User not found"}), 404
                user_id = row[0]
                code = generate_code("ver", digits=6)
                expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                cur.execute("INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (%s, %s, %s, %s)", (user_id, code, "signup", expires))
        send_email(email, code)
        return jsonify({"message": "Verification code resent"}), 200
    except Exception as e:
        logging.exception("Resend verification failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "identifier", "password")
    if not ok: return jsonify({"error": err}), 400
    identifier = data["identifier"].strip()
    password = data["password"]
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, password_hash, is_verified FROM users WHERE email=%s OR username=%s", (identifier, identifier))
                row = cur.fetchone()
        if not row: return jsonify({"error": "User not found"}), 404
        user_id, password_hash, is_verified = row
        if not is_verified: return jsonify({"error": "Verify email first"}), 403
        if not check_password_hash(password_hash, password): return jsonify({"error": "Wrong password"}), 401
        return jsonify({"message": "Login successful", "user_id": user_id}), 200
    except Exception as e:
        logging.exception("Login failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/me", methods=["GET"])
def me():
    user_id = request.args.get("user_id")
    if not user_id: return jsonify({"error": "Missing user_id"}), 400
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, email, is_verified FROM users WHERE id=%s", (user_id,))
                user = cur.fetchone()
                if not user: return jsonify({"error": "User not found"}), 404
                cur.execute("SELECT id, code, purpose, expires_at FROM email_verifications WHERE user_id=%s ORDER BY id DESC LIMIT 10", (user_id,))
                history = cur.fetchall()
        return jsonify({"user": {"id": user[0], "username": user[1], "email": user[2], "is_verified": user[3]}, "history": [{"id": h[0], "code": h[1], "purpose": h[2], "expires_at": h[3].isoformat() if h[3] else None} for h in history]}), 200
    except Exception as e:
        logging.exception("Me endpoint failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/request-reset", methods=["POST"])
def request_reset():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email")
    if not ok: return jsonify({"error": err}), 400
    email = data["email"].strip().lower()
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "User not found"}), 404
                user_id = row[0]
                code = generate_code("pass", digits=6)
                expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                cur.execute("INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (%s, %s, %s, %s)", (user_id, code, "reset_password", expires))
        send_email(email, code)
        return jsonify({"message": "Reset code sent if email delivery succeeded."}), 200
    except Exception as e:
        logging.exception("Request reset failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email", "code", "new_password")
    if not ok: return jsonify({"error": err}), 400
    email = data["email"].strip().lower()
    code = data["code"].strip()
    new_password = data["new_password"]
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row: return jsonify({"error": "User not found"}), 404
                user_id = row[0]
                cur.execute("SELECT id, expires_at FROM email_verifications WHERE user_id=%s AND code=%s AND purpose=%s ORDER BY id DESC LIMIT 1", (user_id, code, "reset_password"))
                ver = cur.fetchone()
                if not ver: return jsonify({"error": "Invalid code"}), 400
                if ver[1] < datetime.datetime.utcnow(): return jsonify({"error": "Code expired"}), 400
                new_hash = generate_password_hash(new_password)
                cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, user_id))
                cur.execute("DELETE FROM email_verifications WHERE user_id=%s AND purpose=%s", (user_id, "reset_password"))
        return jsonify({"message": "Password updated"}), 200
    except Exception as e:
        logging.exception("Reset password failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/_send-test-email", methods=["POST"])
def send_test_email():
    if not TEST_EMAIL_KEY: return jsonify({"error": "Not available"}), 404
    key = request.headers.get("X-TEST-KEY") or request.args.get("key")
    if key!= TEST_EMAIL_KEY: return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    to_email = data.get("email")
    if not to_email: return jsonify({"error": "Missing email"}), 400
    code = generate_code("test", digits=6)
    ok = send_email(to_email, code)
    if ok: return jsonify({"message": "Test email sent"}), 200
    else: return jsonify({"error": "Failed to send test email"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
