import os
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2 import pool

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Environment Variables
DATABASE_URL = os.environ.get("DATABASE_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# PostgreSQL Connection Pooling
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


# --- Helper Search Functions ---

def search_serpapi_ai(query):
    if not SERPAPI_KEY:
        return None, []
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        res = requests.get(url, params=params, timeout=8)
        if res.status_code == 200:
            data = res.json()
            organic_results = data.get("organic_results", [])
            snippets = []
            sources = []
            for item in organic_results[:3]:
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if snippet:
                    snippets.append(snippet)
                if link:
                    sources.append(link)
            return "\n".join(snippets), sources
    except Exception as e:
        logging.warning(f"SerpApi lookup failed: {e}")
    return None, []

def search_tavily(query):
    if not TAVILY_API_KEY:
        return ""
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic"
        }
        res = requests.post(url, json=payload, timeout=8)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            snippets = [r.get("content", "") for r in results[:3] if r.get("content")]
            return "\n".join(snippets)
    except Exception as e:
        logging.warning(f"Tavily lookup failed: {e}")
    return ""


# --- API Endpoints ---

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Tempaper API is running."}), 200


@app.route("/me", methods=["GET"])
def get_me():
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    # Prevent PostgreSQL integer conversion crash when 'demo_user' is sent
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

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not configured on server"}), 500

    # Step 1: Search context with network timeouts
    context, sources = search_serpapi_ai(prompt)
    if not context:
        context = search_tavily(prompt)

    system_prompt = (
        "You are an expert academic researcher writing a clear, well-structured term paper. "
        "Write in a natural, direct academic tone without clichés or filler words."
    )

    user_content = (
        f"Topic/Prompt: {prompt}\n\nReference Material:\n{context}"
        if context
        else prompt
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tempaper.onrender.com",
        "X-Title": "Tempaper Generator",
    }

    # Model endpoints prioritized by speed and free access
    candidate_models = [
        "openrouter/free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-2-9b-it:free",
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
                timeout=12  # Strict 12s timeout per request to avoid Gunicorn worker kill
            )

            if res.status_code == 200:
                response_json = res.json()
                ai_text = response_json["choices"][0]["message"]["content"]
                break
            else:
                last_error = res.text
                logging.warning(
                    f"OpenRouter attempt with {model_name} failed ({res.status_code}): {res.text}"
                )
        except Exception as e:
            logging.warning(f"OpenRouter timeout/error with model {model_name}: {e}")

    if ai_text:
        return jsonify({"result": ai_text, "sources": sources}), 200
    else:
        logging.error(f"All OpenRouter attempts failed: {last_error}")
        return jsonify({"error": "Failed to generate paper from AI model. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
