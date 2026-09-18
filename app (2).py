import os
import random
import string
import logging
from datetime import datetime, timedelta

import requests
import psycopg2
from psycopg2 import pool
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------------------------
# App configuration
# -------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv(
    "RESEND_FROM_EMAIL",
    "onboarding@resend.dev"
)

# Use a model that is commonly available.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# -------------------------------------------------------------------
# Gemini initialization
# -------------------------------------------------------------------

ai_client = None
genai_types = None

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types as genai_types

        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini client initialized successfully.")
    except Exception as error:
        logging.exception("Gemini initialization failed: %s", error)
else:
    logging.warning("GEMINI_API_KEY is not configured.")

# -------------------------------------------------------------------
# Database initialization
# -------------------------------------------------------------------

db_pool = None

if DATABASE_URL:
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            1,
            10,
            DATABASE_URL
        )
        logging.info("Database connection pool created successfully.")
    except Exception as error:
        logging.exception("Database connection pool failed: %s", error)
else:
    logging.warning("DATABASE_URL is not configured.")


def get_db():
    if db_pool is None:
        raise RuntimeError("Database pool is not initialized.")
    return db_pool.getconn()


def release_db(connection):
    if db_pool is not None and connection is not None:
        db_pool.putconn(connection)


def init_db():
    if db_pool is None:
        logging.warning("Database was not initialized. Skipping table creation.")
        return

    connection = None

    try:
        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
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
                """
            )

        connection.commit()
        logging.info("Database schema initialized.")

    except Exception as error:
        if connection:
            connection.rollback()
        logging.exception("Database initialization failed: %s", error)

    finally:
        release_db(connection)


init_db()

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def generate_otp(prefix):
    digits = "".join(random.choices(string.digits, k=3))
    return f"{prefix}{digits}"


def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY:
        logging.warning("RESEND_API_KEY is missing. Email was skipped.")
        return False

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            },
            timeout=15,
        )

        response.raise_for_status()
        return True

    except Exception as error:
        logging.exception("Email sending failed: %s", error)
        return False


def search_tavily(query):
    if not TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY is not configured.")
        return "", []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
            },
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        snippets = [
            item.get("content", "")
            for item in results
            if item.get("content")
        ]

        sources = [
            item.get("url", "")
            for item in results
            if item.get("url")
        ]

        return "\n\n".join(snippets), sources

    except Exception as error:
        logging.warning("Tavily search failed: %s", error)
        return "", []


def call_gemini(prompt, system_instruction):
    if ai_client is None or genai_types is None:
        raise RuntimeError(
            "Gemini is not configured. Add GEMINI_API_KEY to Render."
        )

    response = ai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=5000,
        ),
    )

    result = getattr(response, "text", None)

    if not result:
        raise RuntimeError("Gemini returned an empty response.")

    return result


def get_request_json():
    return request.get_json(silent=True) or {}


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------


@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "Tempaper API is running.",
            "gemini_configured": ai_client is not None,
        }
    ), 200


# -------------------------------------------------------------------
# Authentication
# -------------------------------------------------------------------


@app.route("/register", methods=["POST"])
def register():
    data = get_request_json()

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    if len(password) < 6:
        return jsonify(
            {"error": "Password must be at least 6 characters"}
        ), 400

    connection = None

    try:
        password_hash = generate_password_hash(password)
        otp = generate_otp("ver")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users
                    (username, email, password_hash, is_verified,
                     otp_code, otp_expires_at)
                VALUES (%s, %s, %s, FALSE, %s, %s)
                RETURNING id;
                """,
                (
                    username,
                    email,
                    password_hash,
                    otp,
                    expires_at,
                ),
            )

            user_id = cursor.fetchone()[0]

        connection.commit()

        send_email(
            email,
            "Verify Your Tempaper Account",
            f"""
            <h2>Welcome to Tempaper!</h2>
            <p>Your verification code is:</p>
            <h1>{otp}</h1>
            <p>This code expires in 15 minutes.</p>
            """,
        )

        return jsonify(
            {
                "message": "Registration successful. Check your email.",
                "user_id": user_id,
            }
        ), 201

    except psycopg2.IntegrityError:
        if connection:
            connection.rollback()

        return jsonify(
            {"error": "Username or email already exists"}
        ), 400

    except Exception as error:
        if connection:
            connection.rollback()

        logging.exception("Registration failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/login", methods=["POST"])
def login():
    data = get_request_json()

    identifier = str(data.get("identifier", "")).strip().lower()
    password = str(data.get("password", ""))

    if not identifier or not password:
        return jsonify(
            {"error": "Email/username and password are required"}
        ), 400

    connection = None

    try:
        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, email, password_hash, is_verified
                FROM users
                WHERE LOWER(email) = %s
                   OR LOWER(username) = %s
                """,
                (identifier, identifier),
            )

            user = cursor.fetchone()

        if not user:
            return jsonify(
                {"error": "Invalid username/email or password"}
            ), 401

        if not check_password_hash(user[3], password):
            return jsonify(
                {"error": "Invalid username/email or password"}
            ), 401

        if not user[4]:
            return jsonify(
                {"error": "Please verify your email first"}
            ), 403

        return jsonify(
            {
                "message": "Login successful",
                "user_id": user[0],
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                },
            }
        ), 200

    except Exception as error:
        logging.exception("Login failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/verify-email", methods=["POST"])
def verify_email():
    data = get_request_json()

    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()

    if not email or not code:
        return jsonify({"error": "Email and code are required"}), 400

    connection = None

    try:
        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, otp_code, otp_expires_at
                FROM users
                WHERE LOWER(email) = %s
                """,
                (email,),
            )

            user = cursor.fetchone()

            if not user:
                return jsonify({"error": "User not found"}), 404

            user_id, saved_code, expires_at = user

            if saved_code != code:
                return jsonify(
                    {"error": "Invalid verification code"}
                ), 400

            if expires_at and datetime.utcnow() > expires_at:
                return jsonify(
                    {"error": "Verification code has expired"}
                ), 400

            cursor.execute(
                """
                UPDATE users
                SET is_verified = TRUE,
                    otp_code = NULL,
                    otp_expires_at = NULL
                WHERE id = %s
                """,
                (user_id,),
            )

        connection.commit()

        return jsonify(
            {"message": "Account verified successfully"}
        ), 200

    except Exception as error:
        if connection:
            connection.rollback()

        logging.exception("Email verification failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    data = get_request_json()
    email = str(data.get("email", "")).strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    connection = None

    try:
        otp = generate_otp("ver")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET otp_code = %s,
                    otp_expires_at = %s
                WHERE LOWER(email) = %s
                RETURNING is_verified
                """,
                (otp, expires_at, email),
            )

            result = cursor.fetchone()

            if not result:
                return jsonify({"error": "Email not found"}), 404

            if result[0]:
                return jsonify(
                    {"message": "Account is already verified"}
                ), 200

        connection.commit()

        send_email(
            email,
            "Your Tempaper Verification Code",
            f"""
            <p>Your new verification code is:</p>
            <h1>{otp}</h1>
            <p>This code expires in 15 minutes.</p>
            """,
        )

        return jsonify(
            {"message": "Verification code resent successfully"}
        ), 200

    except Exception as error:
        if connection:
            connection.rollback()

        logging.exception("Resend verification failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/request-reset", methods=["POST"])
def request_reset():
    data = get_request_json()
    email = str(data.get("email", "")).strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    connection = None

    try:
        otp = generate_otp("pass")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET otp_code = %s,
                    otp_expires_at = %s
                WHERE LOWER(email) = %s
                RETURNING id
                """,
                (otp, expires_at, email),
            )

            result = cursor.fetchone()

        connection.commit()

        if result:
            send_email(
                email,
                "Tempaper Password Reset Code",
                f"""
                <p>Your password reset code is:</p>
                <h1>{otp}</h1>
                <p>This code expires in 15 minutes.</p>
                """,
            )

        return jsonify(
            {"message": "If the email exists, a code was sent."}
        ), 200

    except Exception as error:
        if connection:
            connection.rollback()

        logging.exception("Password reset request failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = get_request_json()

    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    new_password = str(data.get("new_password", ""))

    if not email or not code or not new_password:
        return jsonify({"error": "Missing required fields"}), 400

    if len(new_password) < 6:
        return jsonify(
            {"error": "Password must be at least 6 characters"}
        ), 400

    connection = None

    try:
        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, otp_code, otp_expires_at
                FROM users
                WHERE LOWER(email) = %s
                """,
                (email,),
            )

            user = cursor.fetchone()

            if not user or user[1] != code:
                return jsonify({"error": "Invalid code or email"}), 400

            expires_at = user[2]

            if expires_at and datetime.utcnow() > expires_at:
                return jsonify(
                    {"error": "Reset code has expired"}
                ), 400

            cursor.execute(
                """
                UPDATE users
                SET password_hash = %s,
                    otp_code = NULL,
                    otp_expires_at = NULL
                WHERE id = %s
                """,
                (
                    generate_password_hash(new_password),
                    user[0],
                ),
            )

        connection.commit()

        return jsonify(
            {"message": "Password updated successfully"}
        ), 200

    except Exception as error:
        if connection:
            connection.rollback()

        logging.exception("Password reset failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


@app.route("/me", methods=["GET"])
def get_me():
    user_id = request.args.get("user_id", "").strip()

    if not user_id or not user_id.isdigit():
        return jsonify(
            {
                "user": {
                    "id": 0,
                    "username": "Guest User",
                    "email": "guest@tempaper.com",
                    "is_verified": False,
                }
            }
        ), 200

    connection = None

    try:
        connection = get_db()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, email, is_verified
                FROM users
                WHERE id = %s
                """,
                (int(user_id),),
            )

            user = cursor.fetchone()

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify(
            {
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "email": user[2],
                    "is_verified": user[3],
                }
            }
        ), 200

    except Exception as error:
        logging.exception("Profile lookup failed: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    finally:
        release_db(connection)


# -------------------------------------------------------------------
# AI endpoints
# -------------------------------------------------------------------


@app.route("/chat", methods=["POST"])
def assistant_chat():
    data = get_request_json()
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if ai_client is None:
        return jsonify(
            {
                "error": "GEMINI_API_KEY is not configured on the server"
            }
        ), 500

    try:
        result = call_gemini(
            prompt,
            """
            You are Tempaper Copilot, an academic research assistant.
            Give clear, useful, concise answers.
            Use headings, bullet points, and examples when helpful.
            Do not invent citations.
            """,
        )

        return jsonify({"result": result}), 200

    except Exception as error:
        logging.exception("Chat generation failed: %s", error)
        return jsonify(
            {"error": "Assistant failed to generate a response"}
        ), 500


@app.route("/generate", methods=["POST"])
def generate_paper():
    data = get_request_json()
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if ai_client is None:
        return jsonify(
            {
                "error": "GEMINI_API_KEY is not configured on the server"
            }
        ), 500

    try:
        research_text, sources = search_tavily(prompt)

        full_prompt = prompt

        if research_text:
            full_prompt = f"""
Topic:
{prompt}

Research material:
{research_text}

Use the research material as background information.
Do not copy it word for word.
"""

        system_instruction = """
You are a professional university lecturer and academic writer.

Generate a complete academic term paper in Markdown.

Requirements:

1. Start with a clear title.
2. Include an abstract.
3. Include a table of contents.
4. Include an introduction.
5. Include 3 to 5 detailed body sections.
6. Include examples and analysis.
7. Include a conclusion.
8. Include recommendations.
9. Include a references section.
10. Write between 1200 and 2000 words.
11. Use Markdown headings:
    # for the title
    ## for main sections
    ### for subsections
12. Use [1], [2], and similar citations only when supported by the
    provided research material.
13. Do not return JSON.
14. Return only the paper content.
"""

        result = call_gemini(
            full_prompt,
            system_instruction
        )

        # Return both names so old and new dashboard versions work.
        return jsonify(
            {
                "success": True,
                "result": result,
                "paper": result,
                "sources": sources,
            }
        ), 200

    except Exception as error:
        logging.exception("Paper generation failed: %s", error)
        return jsonify(
            {
                "success": False,
                "error": "Failed to generate paper"
            }
        ), 500


# -------------------------------------------------------------------
# Start server
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )