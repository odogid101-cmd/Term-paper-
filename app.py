import datetime
import logging
import os
import secrets
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.errors
import requests
from serpapi import GoogleSearch
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PORT = int(os.getenv("PORT", 5000))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")


def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        is_verified BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS email_verifications (
                        id SERIAL PRIMARY KEY,
                        user_id INT REFERENCES users(id) ON DELETE CASCADE,
                        code VARCHAR(10) NOT NULL,
                        purpose VARCHAR(50) NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )
                conn.commit()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.exception("Database initialization failed: %s", e)


init_db()


def generate_code(digits=6):
    return f"{secrets.randbelow(10**digits):0{digits}d}"


def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY or not FROM_EMAIL:
        logging.error("Missing RESEND_API_KEY or FROM_EMAIL configuration.")
        return False
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            },
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        logging.exception("Failed to send email: %s", e)
        return False


def search_serpapi_ai(query: str):
    """Fetches Google AI Mode summaries and reference links via SerpApi."""
    if not SERPAPI_API_KEY:
        return "", []

    try:
        params = {
            "engine": "google_ai_mode",
            "q": query,
            "api_key": SERPAPI_API_KEY,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        extracted_text = []
        sources = []

        text_blocks = results.get("text_blocks", [])
        for block in text_blocks:
            if isinstance(block, dict) and "snippet" in block:
                extracted_text.append(block["snippet"])
            elif isinstance(block, str):
                extracted_text.append(block)

        references = results.get("references", [])
        for ref in references:
            sources.append(
                {"title": ref.get("title", ""), "link": ref.get("link", "")}
            )

        context = "\n".join(extracted_text)
        return context, sources
    except Exception as e:
        logging.error(f"SerpApi error: {e}")
        return "", []


def search_tavily(query: str):
    """Fallback search using Tavily API."""
    if not TAVILY_API_KEY:
        return ""

    try:
        res = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 3,
            },
            timeout=8,
        )
        if res.status_code == 200:
            results = res.json().get("results", [])
            return "\n".join(
                [f"- {r.get('title')}: {r.get('content')}" for r in results]
            )
    except Exception as e:
        logging.error(f"Tavily search failed: {e}")
    return ""


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is online"}), 200


@app.route("/generate", methods=["POST"])
def generate_paper():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not OPENROUTER_API_KEY:
        return (
            jsonify(
                {"error": "OPENROUTER_API_KEY is not configured on server"}
            ),
            500,
        )

    # Fetch context from SerpApi (Google AI Mode)
    context, sources = search_serpapi_ai(prompt)

    # Fallback to Tavily if SerpApi context is empty
    if not context:
        context = search_tavily(prompt)

    system_prompt = (
        "You are an expert academic researcher writing a clear, well-structured term paper. "
        "Guidelines:\n"
        "1. Write in a natural, direct, human academic tone.\n"
        "2. Strictly AVOID AI clichés/buzzwords like 'delve', 'tapestry', 'testament', 'pivotal', 'in conclusion', or 'furthermore'.\n"
        "3. Incorporate provided reference context smoothly into the text.\n"
        "4. Vary your sentence structures."
    )

    user_content = (
        f"Topic/Prompt: {prompt}\n\nReference Material:\n{context}"
        if context
        else prompt
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Term Paper Assistant",
    }

    # Free candidate models on OpenRouter
    candidate_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-2-9b-it:free",
        "openrouter/free",
    ]

    ai_text = None
    last_error = ""

    for model_name in candidate_models:
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.7,
                },
                timeout=30,
            )

            if res.status_code == 200:
                ai_text = res.json()["choices"][0]["message"]["content"]
                break
            else:
                last_error = res.text
                logging.warning(
                    f"OpenRouter attempt with {model_name} failed ({res.status_code}): {res.text}"
                )
        except Exception as e:
            logging.exception(
                f"OpenRouter error with model {model_name}: %s", e
            )

    if ai_text:
        return jsonify({"result": ai_text, "sources": sources}), 200
    else:
        logging.error(f"All OpenRouter attempts failed: {last_error}")
        return jsonify({"error": "Failed to generate paper from AI model"}), 500


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
                    (username, email, pwd_hash),
                )
                user_id = cur.fetchone()[0]

                code = generate_code(digits=6)
                expires_at = datetime.datetime.utcnow() + datetime.timedelta(
                    minutes=15
                )
                cur.execute(
                    "INSERT INTO email_verifications (user_id, code, purpose, expires_at) VALUES (%s, %s, %s, %s)",
                    (user_id, code, "verify", expires_at),
                )
                conn.commit()

        email_html = f"<h3>Welcome to Tempaper!</h3><p>Your verification code is: <b>{code}</b></p>"
        send_email(email, "Verify your Tempaper account", email_html)

        return (
            jsonify(
                {
                    "message": "Registration successful. Please check your email for verification code.",
                    "user_id": user_id,
                }
            ),
            201,
        )

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
                    (identifier.lower(), identifier),
                )
                row = cur.fetchone()

                if not row or not check_password_hash(row[3], password):
                    return jsonify({"error": "Invalid credentials"}), 401

                user_id, username, email, _, is_verified = row

                if not is_verified:
                    return (
                        jsonify(
                            {"error": "Verify email first", "user_id": user_id}
                        ),
                        403,
                    )

                return (
                    jsonify(
                        {
                            "message": "Login successful",
                            "user_id": user_id,
                            "user": {
                                "id": user_id,
                                "username": username,
                                "email": email,
                            },
                        }
                    ),
                    200,
                )

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
                cur.execute(
                    "SELECT id, username, email, is_verified FROM users WHERE id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "User not found"}), 404

                return (
                    jsonify(
                        {
                            "user": {
                                "id": row[0],
                                "username": row[1],
                                "email": row[2],
                                "is_verified": row[3],
                            }
                        }
                    ),
                    200,
                )
    except Exception as e:
        logging.exception("Get profile failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
