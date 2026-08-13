# app.py (API-only)
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import psycopg2.errors
import os
import secrets
import datetime
import requests
import logging

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
    logging.warning("RESEND_API_KEY and/or FROM_EMAIL not set — send_email will fail until provided")

def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        is_verified BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS email_verifications (
                        id SERIAL PRIMARY KEY,
                        user_id INT REFERENCES users(id) ON DELETE CASCADE,
                        code VARCHAR(10) NOT NULL,
                        purpose VARCHAR(50) NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.exception("Database initialization failed: %s", e)

init_db()

def generate_code(purpose="verify", digits=6):
    if digits == 6:
        return f"{secrets.randbelow(1000000):06d}"
    return secrets.token_hex(4)

def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY or not FROM_EMAIL:
        logging.error("Cannot send email: RESEND_API_KEY or FROM_EMAIL is missing.")
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
                "to": [to_email],
                "subject": subject,
                "html": html_content
            },
            timeout=10
        )
        if resp.status_code in (200, 201):
            logging.info("Email sent successfully to %s", to_email)
            return True
        else:
            logging.error("Resend API error (%s): %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        logging.exception("Failed to send email to %s: %s", to_email, e)
        return False

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is online"}), 200

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    pwd_hash = generate_password_hash(password)

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, email, password_hash, is_verified) VALUES (%s, %s, %s, FALSE) RETURNING id",
                    (username, email, pwd_hash)
                )
                user_id = cur.fetchone()[0]

                code = generate_code("verify", digits=6)
                expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
                cur.execute(
                    "INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (%s, %s, %s, %s)",
                    (user_id, code, "verify", expires_at)
                )
                conn.commit()

        email_html = f"<h3>Welcome to Tempaper!</h3><p>Your verification code is: <b>{code}</b></p>"
        send_email(email, "Verify your Tempaper account", email_html)

        return jsonify({
            "message": "Registration successful. Please check your email for verification code.",
            "user_id": user_id
        }), 201

    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "Username or email already exists"}), 400
    except Exception as e:
        logging.exception("Registration failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("identifier") or "").strip()
    password = data.get("password") or ""

    if not identifier or not password:
        return jsonify({"error": "Missing identifier or password"}), 400

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, email, password_hash, is_verified FROM users WHERE email = %s OR username = %s",
                    (identifier.lower(), identifier)
                )
                row = cur.fetchone()

                if not row or not check_password_hash(row[3], password):
                    return jsonify({"error": "Invalid credentials"}), 401

                user_id, username, email, _, is_verified = row

                if not is_verified:
                    return jsonify({"error": "Verify email first", "user_id": user_id}), 403

                return jsonify({
                    "message": "Login successful",
                    "user_id": user_id,
                    "user": {"id": user_id, "username": username, "email": email}
                }), 200

    except Exception as e:
        logging.exception("Login failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/me", methods=["GET"])
def get_me():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, email, is_verified FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "User not found"}), 404

                return jsonify({
                    "user": {
                        "id": row[0],
                        "username": row[1],
                        "email": row[2],
                        "is_verified": row[3]
                    }
                }), 200
    except Exception as e:
        logging.exception("Get profile failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
