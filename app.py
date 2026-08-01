from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import psycopg2.errors
import os
import secrets
import datetime
import requests
import logging
from typing import Optional

# -------------------------
# App & logging
# -------------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# -------------------------
# Config (set in environment)
# -------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
PORT = int(os.getenv("PORT", 5000))
# Optional test key for /_send-test-email route
TEST_EMAIL_KEY = os.getenv("TEST_EMAIL_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

if not RESEND_API_KEY or not FROM_EMAIL:
    logging.warning("RESEND_API_KEY and/or FROM_EMAIL not set — send_email will fail until they are provided")

# -------------------------
# Database helpers
# -------------------------
def get_db():
    # psycopg2 connection; using sslmode=require matches many managed Postgres providers
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
        # commit on context manager exit

init_db()

# -------------------------
# Utilities
# -------------------------
def generate_code(prefix: str, digits: int = 6) -> str:
    """Generate a short unpredictable code with a prefix."""
    number = secrets.randbelow(10 ** digits)
    return f"{prefix}{number:0{digits}d}"

def send_email(to_email: str, code: str) -> bool:
    """Send email via Resend. Returns True on success. Logs response details for debugging."""
    if not RESEND_API_KEY or not FROM_EMAIL:
        logging.error("Email not sent: RESEND_API_KEY or FROM_EMAIL missing")
        return False

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "from": FROM_EMAIL,
                "to": to_email,
                "subject": "Your verification code",
                "html": f"<h2>{code}</h2><p>This code expires in 10 minutes.</p>"
            },
            timeout=10
        )
        logging.info("Resend response status: %s", resp.status_code)
        logging.info("Resend response body: %s", resp.text)
        resp.raise_for_status()
        return True
    except requests.exceptions.HTTPError as he:
        logging.exception("Resend HTTP error: %s", he)
        return False
    except Exception as e:
        logging.exception("Failed to send email: %s", e)
        return False

def _required_fields(data: dict, *fields):
    missing = [f for f in fields if not data.get(f)]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    return True, None

# -------------------------
# Static pages (serve from static/)
# -------------------------
@app.route("/", methods=["GET"])
def root():
    # serve setup-signin.html from the static folder
    return send_from_directory(app.static_folder, "setup-signin.html")

@app.route("/verify.html", methods=["GET"])
def serve_verify_html():
    return send_from_directory(app.static_folder, "verify.html")

# -------------------------
# Routes (API)
# -------------------------
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
                cur.execute("""
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s) RETURNING id
                """, (username, email, password_hash))
                user_id = cur.fetchone()[0]

                code = generate_code("ver", digits=6)
                expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                cur.execute("""
                INSERT INTO email_verifications (user_id, code, purpose, expires_at)
                VALUES (%s, %s, %s, %s)
                """, (user_id, code, "signup", expires))

        sent = send_email(email, code)
        if not sent:
            logging.warning("Verification email could not be delivered to %s", email)

        return jsonify({"message": "User registered; verification code sent if email delivery succeeded."}), 201

    except psycopg2.errors.UniqueViolation:
        # duplicate username or email
        logging.info("Attempt to register an existing user: %s / %s", username, email)
        return jsonify({"error": "Username or email already exists"}), 409
    except Exception as e:
        logging.exception("Registration failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/verify-email", methods=["POST"])
def verify_email():
    """Verify a signup code. Expects json with 'email' and 'code'."""
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email", "code")
    if not ok:
        return jsonify({"error": err}), 400

    email = data["email"].strip().lower()
    code = data["code"].strip()

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "User not found"}), 404
                user_id = row[0]

                cur.execute("""
                SELECT id, expires_at FROM email_verifications
                WHERE user_id=%s AND code=%s AND purpose=%s
                ORDER BY id DESC LIMIT 1
                """, (user_id, code, "signup"))
                ver = cur.fetchone()
                if not ver:
                    return jsonify({"error": "Invalid code"}), 400

                expires_at = ver[1]
                if expires_at is None or expires_at < datetime.datetime.utcnow():
                    return jsonify({"error": "Code expired"}), 400

                cur.execute("UPDATE users SET is_verified = TRUE WHERE id = %s", (user_id,))
                cur.execute("DELETE FROM email_verifications WHERE user_id=%s AND purpose=%s", (user_id, "signup"))

        return jsonify({"message": "Email verified"}), 200

    except Exception as e:
        logging.exception("Email verification failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "identifier", "password")
    if not ok:
        return jsonify({"error": err}), 400

    identifier = data["identifier"].strip()
    password = data["password"]

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT id, password_hash, is_verified
                FROM users WHERE email=%s OR username=%s
                """, (identifier, identifier))
                row = cur.fetchone()

        if not row:
            return jsonify({"error": "User not found"}), 404

        user_id, password_hash, is_verified = row
        if not is_verified:
            return jsonify({"error": "Verify email first"}), 403

        if not check_password_hash(password_hash, password):
            return jsonify({"error": "Wrong password"}), 401

        # TODO: return a session token / JWT here
        return jsonify({"message": "Login successful", "user_id": user_id}), 200

    except Exception as e:
        logging.exception("Login failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/request-reset", methods=["POST"])
def request_reset():
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email")
    if not ok:
        return jsonify({"error": err}), 400

    email = data["email"].strip().lower()

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "User not found"}), 404
                user_id = row[0]

                code = generate_code("pass", digits=6)
                expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                cur.execute("""
                INSERT INTO email_verifications (user_id, code, purpose, expires_at)
                VALUES (%s, %s, %s, %s)
                """, (user_id, code, "reset_password", expires))

        sent = send_email(email, code)
        if not sent:
            logging.warning("Reset email could not be delivered to %s", email)

        return jsonify({"message": "Reset code sent if email delivery succeeded."}), 200

    except Exception as e:
        logging.exception("Request reset failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/reset-password", methods=["POST"])
def reset_password():
    """
    Expect JSON with: email, code, new_password
    Validates the reset code before changing the password.
    """
    data = request.get_json(silent=True) or {}
    ok, err = _required_fields(data, "email", "code", "new_password")
    if not ok:
        return jsonify({"error": err}), 400

    email = data["email"].strip().lower()
    code = data["code"].strip()
    new_password = data["new_password"]

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "User not found"}), 404
                user_id = row[0]

                cur.execute("""
                SELECT id, expires_at FROM email_verifications
                WHERE user_id=%s AND code=%s AND purpose=%s
                ORDER BY id DESC LIMIT 1
                """, (user_id, code, "reset_password"))
                ver = cur.fetchone()
                if not ver:
                    return jsonify({"error": "Invalid code"}), 400

                expires_at = ver[1]
                if expires_at is None or expires_at < datetime.datetime.utcnow():
                    return jsonify({"error": "Code expired"}), 400

                new_hash = generate_password_hash(new_password)
                cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, user_id))
                cur.execute("DELETE FROM email_verifications WHERE user_id=%s AND purpose=%s", (user_id, "reset_password"))

        return jsonify({"message": "Password updated"}), 200

    except Exception as e:
        logging.exception("Reset password failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

# -------------------------
# Optional: test email endpoint (safe-guarded)
# -------------------------
@app.route("/_send-test-email", methods=["POST"])
def send_test_email():
    """Send a test email. Protect by TEST_EMAIL_KEY env var (set this to call)."""
    if not TEST_EMAIL_KEY:
        return jsonify({"error": "Not available"}), 404

    key = request.headers.get("X-TEST-KEY") or request.args.get("key")
    if key != TEST_EMAIL_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    to_email = data.get("email")
    if not to_email:
        return jsonify({"error": "Missing email"}), 400

    code = generate_code("test", digits=6)
    ok = send_email(to_email, code)
    if ok:
        return jsonify({"message": "Test email sent"}), 200
    else:
        return jsonify({"error": "Failed to send test email"}), 500

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Use 0.0.0.0 so the container is reachable; PORT can be provided by the environment
    app.run(host="0.0.0.0", port=PORT)
