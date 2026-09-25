import os
import random
import string
import logging
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2 import pool
import requests
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# Google Gemini initialization
ai_client = None
gemini_types = None
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY is missing.")
else:
    try:
        from google import genai
        from google.genai import types

        gemini_types = types
        ai_client = genai.Client(api_key=GEMINI_API_KEY)

        logging.info(
            "Gemini client initialized successfully. Model: %s",
            GEMINI_MODEL
        )
    except Exception as error:
        logging.exception("Gemini client initialization failed: %s", error)
        ai_client = None
        gemini_types = None

db_pool = None
if DATABASE_URL:
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, DATABASE_URL)
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.error(f"Error creating database connection pool: {e}")

def get_db():
    if not db_pool:
        raise Exception("Database pool is not initialized.")
    return db_pool.getconn()

def release_db(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

def init_db():
    if not db_pool:
        logging.error("Cannot initialize DB: db_pool is not ready.")
        return

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_verified BOOLEAN DEFAULT FALSE,
                    otp_code VARCHAR(20),
                    otp_expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logging.info("Database schema initialized successfully.")
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error initializing database schema: {e}")
    finally:
        if conn:
            release_db(conn)

init_db()

def generate_otp(prefix):
    digits = "".join(random.choices(string.digits, k=3))
    return f"{prefix}{digits}"

def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY:
        logging.warning("RESEND_API_KEY is not configured. Email skipped.")
        return False

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [to_email],
        "subject": subject,
        "html": html_content
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        logging.info("Resend response: status=%s body=%s", res.status_code, res.text)
        return res.status_code in (200, 201)
    except Exception as e:
        logging.exception("Failed to send email via Resend: %s", e)
        return False

def search_tavily(query):
    if not TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY is not configured.")
        return "", []
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        res = requests.post(url, json=payload, timeout=8)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            snippets = [r.get("content", "") for r in results if r.get("content")]
            sources = [r.get("url", "") for r in results if r.get("url")]
            return "\n\n".join(snippets), sources
    except Exception as e:
        logging.warning(f"Tavily lookup failed: {e}")
    return "", []

def generate_with_gemini(prompt, context_text=""):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing from the server environment.")

    if not ai_client or not gemini_types:
        raise RuntimeError(
            "Gemini client is not initialized. Check google-genai installation and GEMINI_API_KEY."
        )

    final_prompt = context_text if context_text else prompt

    system_instruction = """
You are Tempaper, an expert academic writing assistant.

Write a clear, well-structured academic paper based on the user's request.

Requirements:
- Use a professional academic tone.
- Include a suitable title.
- Use Markdown headings with #, ##, and ###.
- Include an introduction, body sections, and conclusion when appropriate.
- Use clear paragraphs.
- Do not return placeholder text.
- Do not claim to have searched academic databases unless reference material is provided.
- Do not invent citations, statistics, quotations, or sources.
"""

    response = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=final_prompt,
        config=gemini_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=8192
        )
    )

    generated_text = getattr(response, "text", None)
    if not generated_text or not generated_text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return generated_text.strip()


@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ok", "message": "Tempaper API is running."}), 200


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    hashed = generate_password_hash(password)
    otp = generate_otp("ver")
    expires_at = datetime.utcnow() + timedelta(minutes=15)

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, is_verified, otp_code, otp_expires_at)
                VALUES (%s, %s, %s, FALSE, %s, %s)
                RETURNING id;
                """,
                (username, email, hashed, otp, expires_at)
            )
            user_id = cur.fetchone()[0]
            conn.commit()

        email_sent = send_email(
            email,
            "Verify Your Tempaper Account",
            f"<p>Your verification code is: <strong>{otp}</strong></p><p>Expires in 15 minutes.</p>"
        )

        if not email_sent:
            logging.error("Registration email failed for %s", email)
            return jsonify({"error": "Could not send verification email. Please try again later."}), 502

        return jsonify({"message": "Registration successful. Check your email for code.", "user_id": user_id}), 201
    except psycopg2.IntegrityError:
        if conn:
            conn.rollback()
        return jsonify({"error": "Username or Email already exists"}), 400
    except Exception as e:
        if conn:
            conn.rollback()
        logging.exception("Registration failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier", "").strip().lower()
    password = data.get("password", "")

    if not identifier or not password:
        return jsonify({"error": "Missing email/username or password"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, is_verified FROM users WHERE LOWER(email) = %s OR LOWER(username) = %s",
                (identifier, identifier)
            )
            row = cur.fetchone()
            if not row or not check_password_hash(row[3], password):
                return jsonify({"error": "Invalid username/email or password"}), 401

            if not row[4]:
                return jsonify({"error": "Verify email first"}), 403

            return jsonify({
                "message": "Login successful",
                "user_id": row[0],
                "user": {"id": row[0], "username": row[1], "email": row[2]}
            }), 200
    except Exception as e:
        logging.exception("Login error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)


@app.route("/verify-email", methods=["POST"])
def verify_email():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()

    if not email or not code:
        return jsonify({"error": "Email and code are required"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, otp_code, otp_expires_at FROM users WHERE LOWER(email) = %s",
                (email,)
            )
            row = cur.fetchone()

            if not row:
                return jsonify({"error": "User not found"}), 404

            user_id, saved_otp, expires_at = row

            if saved_otp != code:
                return jsonify({"error": "Invalid verification code"}), 400

            if expires_at and datetime.utcnow() > expires_at:
                return jsonify({"error": "Verification code has expired"}), 400

            cur.execute(
                "UPDATE users SET is_verified = TRUE, otp_code = NULL, otp_expires_at = NULL WHERE id = %s",
                (user_id,)
            )
            conn.commit()

            return jsonify({"message": "Account verified successfully!"}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        logging.exception("Verify error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)


@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = generate_otp("ver")
    expires_at = datetime.utcnow() + timedelta(minutes=15)

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET otp_code = %s, otp_expires_at = %s WHERE LOWER(email) = %s RETURNING is_verified",
                (otp, expires_at, email)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Email not found"}), 404

            if row[0]:
                return jsonify({"message": "Account is already verified"}), 200

            conn.commit()

        email_sent = send_email(
            email,
            "Your Verification Code",
            f"<p>Your new verification code is: <strong>{otp}</strong></p>"
        )

        if not email_sent:
            return jsonify({"error": "Verification email could not be sent."}), 502

        return jsonify({"message": "Verification code resent successfully"}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        logging.exception("Resend verification failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)


@app.route("/request-reset", methods=["POST"])
def request_reset():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = generate_otp("pass")
    expires_at = datetime.utcnow() + timedelta(minutes=15)

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET otp_code = %s, otp_expires_at = %s WHERE LOWER(email) = %s RETURNING id",
                (otp, expires_at, email)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"message": "If the email exists, a code was sent."}), 200

            conn.commit()

        email_sent = send_email(
            email,
            "Password Reset Code",
            f"<div><h2>Tempaper Password Reset</h2><p>Your password reset code is:</p><h1>{otp}</h1><p>This code expires in 15 minutes.</p></div>"
        )

        if not email_sent:
            logging.error("Password reset email failed for %s", email)
            return jsonify({"error": "The reset code could not be sent. Please try again later."}), 502

        return jsonify({"message": "Reset code sent to your email."}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        logging.exception("Request reset failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    code = data.get("code", "").strip()
    new_password = data.get("new_password", "")

    if not email or not code or not new_password:
        return jsonify({"error": "Missing required fields"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, otp_code, otp_expires_at FROM users WHERE LOWER(email) = %s",
                (email,)
            )
            row = cur.fetchone()

            if not row or row[1] != code:
                return jsonify({"error": "Invalid code or email"}), 400

            _, _, expires_at = row
            if expires_at and datetime.utcnow() > expires_at:
                return jsonify({"error": "Reset code has expired"}), 400

            new_hashed = generate_password_hash(new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s, otp_code = NULL, otp_expires_at = NULL WHERE LOWER(email) = %s",
                (new_hashed, email)
            )
            conn.commit()

            return jsonify({"message": "Password updated successfully!"}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        logging.exception("Reset password failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/me", methods=["GET"])
def get_me():
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    if not str(user_id).isdigit():
        return jsonify({
            "user": {
                "id": 0,
                "username": "Guest User",
                "email": "guest@tempaper.com",
                "is_verified": False
            }
        }), 200

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, is_verified FROM users WHERE id = %s",
                (int(user_id),)
            )
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
    finally:
        if conn:
            release_db(conn)

@app.route("/generate", methods=["POST"])
def generate_paper():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        tavily_context, sources = search_tavily(prompt)

        if tavily_context:
            final_content = f"Topic/Prompt: {prompt}\n\nReference Material:\n{tavily_context}"
        else:
            final_content = prompt

        result = generate_with_gemini(prompt, final_content)
        return jsonify({
            "result": result,
            "sources": sources
        }), 200
    except Exception as e:
        logging.exception("Generation error: %s", e)
        return jsonify({
            "error": "Gemini failed to generate the paper. Check the server logs for the exact error."
        }), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
