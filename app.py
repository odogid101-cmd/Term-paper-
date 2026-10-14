import os
import random
import string
import time
import logging
import functools
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS

# Try to import real Cache; if backend import fails at init time (as in your logs),
# fall back to a lightweight wrapper using cachelib.SimpleCache to avoid crashing.
cache = None
_use_real_flask_caching = False

try:
    # Import Cache class (may succeed) but instantiation may fail if backend import fails.
    from flask_caching import Cache as FlaskCache
    _use_real_flask_caching = True
except Exception as e:
    # If import of the package itself fails, we will fall back below.
    logging.getLogger(__name__).warning("flask_caching import failed: %s", e)
    _use_real_flask_caching = False

# Initialize Flask app early so we can try init_app on real cache if available
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["https://odogid101-cmd.github.io", "http://localhost:3000", "http://localhost:5000"])

# Now try to set up cache in a safe way
if _use_real_flask_caching:
    try:
        # Attempt to create and init real Flask-Caching. This may raise if backend mapping changed.
        cache = FlaskCache(config={"CACHE_TYPE": "simple", "CACHE_DEFAULT_TIMEOUT": 300})
        cache.init_app(app)
        logging.getLogger(__name__).info("Initialized Flask-Caching (real).")
    except Exception as e:
        logging.getLogger(__name__).warning("Flask-Caching initialization failed: %s. Falling back to simple cache.", e)
        cache = None

# If cache is still None, create a small fallback wrapper around cachelib.SimpleCache
if cache is None:
    try:
        from cachelib import SimpleCache
        _simple_cache_store = SimpleCache(default_timeout=300)

        class FallbackCache:
            def __init__(self, store):
                self._store = store

            def init_app(self, app, config=None):
                # no-op for compatibility
                return

            def get(self, key):
                return self._store.get(key)

            def set(self, key, value, timeout=None):
                return self._store.set(key, value, timeout)

            def cached(self, timeout=300, query_string=False):
                """
                Decorator to mimic Flask-Caching cached decorator for simple uses in this app.
                Caching key is built from request.path and optionally the query string.
                The decorator stores the full return value of the view function (including Flask Response).
                """
                def decorator(f):
                    @functools.wraps(f)
                    def wrapped(*args, **kwargs):
                        try:
                            # Build key from path + optionally query string
                            key = f"cached:v1:{request.path}"
                            if query_string:
                                qs = request.query_string.decode() if request.query_string else ""
                                key = f"{key}?{qs}"
                            cached_val = self._store.get(key)
                            if cached_val is not None:
                                return cached_val
                            rv = f(*args, **kwargs)
                            # Cache the response only if it's not an error or None
                            self._store.set(key, rv, timeout)
                            return rv
                        except Exception as exc:
                            # On any cache-side problem, log and continue without caching
                            logging.getLogger(__name__).exception("Fallback cache error: %s", exc)
                            return f(*args, **kwargs)
                    return wrapped
                return decorator

        cache = FallbackCache(_simple_cache_store)
        logging.getLogger(__name__).info("Using fallback SimpleCache wrapper for caching.")
    except Exception as e:
        # As a last resort provide a no-op cache object with 'cached' decorator that returns the function unchanged
        logging.getLogger(__name__).warning("Failed to initialize fallback cachelib SimpleCache: %s. Using no-op cache.", e)

        class NoOpCache:
            def cached(self, *args, **kwargs):
                def decorator(f):
                    return f
                return decorator

            def init_app(self, app, config=None):
                return

        cache = NoOpCache()

# Now continue with the rest of your app setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import psycopg2
from psycopg2 import pool
import requests
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash-latest")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# Old Gemini SDK for Python 3.6
genai = None
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Generative AI initialized successfully.")
except Exception as e:
    logger.warning("Google Generative AI init failed: %s", e)

db_pool = None
if DATABASE_URL:
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(5, 20, DATABASE_URL)
        logger.info("Database pool created.")
    except Exception as e:
        logger.exception("Error creating database pool: %s", e)

def get_db():
    if not db_pool:
        raise RuntimeError("Database pool is not initialized.")
    return db_pool.getconn()

def release_db(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

def init_db():
    if not db_pool:
        logger.error("DB pool not ready")
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                );
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(LOWER(email));")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(LOWER(username));")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    session_id VARCHAR(64) UNIQUE NOT NULL,
                    title VARCHAR(200),
                    prompt_text TEXT,
                    result_text TEXT,
                    word_count INTEGER DEFAULT 0,
                    page_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    theme VARCHAR(20) DEFAULT 'system',
                    font_size INTEGER DEFAULT 16,
                    notifications BOOLEAN DEFAULT TRUE,
                    auto_save BOOLEAN DEFAULT TRUE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("Database schema initialized.")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Error initializing database schema: %s", e)
    finally:
        if conn:
            release_db(conn)

init_db()

def generate_otp(prefix):
    return f"{prefix}{''.join(random.choices(string.digits, k=3))}"

def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY:
        logger.error("RESEND_API_KEY is not set")
        return False
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            json={"from": RESEND_FROM_EMAIL, "to": [to_email], "subject": subject, "html": html_content},
            headers={"Authorization": "Bearer " + RESEND_API_KEY, "Content-Type": "application/json"},
            timeout=15
        )
        logger.info("Resend response: %s %s", response.status_code, response.text)
        return response.status_code in (200, 201)
    except Exception as e:
        logger.exception("Email send failed: %s", e)
        return False

def search_tavily(query):
    if not TAVILY_API_KEY:
        return "", []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 3},
            timeout=8
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            snippets = [item.get("content", "") for item in results if item.get("content")]
            sources = [item.get("url", "") for item in results if item.get("url")]
            return "\n\n".join(snippets), sources
        logger.warning("Tavily failed: %s %s", response.status_code, response.text)
    except Exception as e:
        logger.warning("Tavily lookup failed: %s", e)
    return "", []
def generate_with_gemini(prompt, context_text=""):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")

    final_prompt = context_text if context_text else prompt
    system_instruction = (
        "You are Tempaper, an expert academic writing assistant.\n"
        "Write a clear, well-structured academic paper based on the user's request.\n"
        "Use a professional academic tone, include a suitable title, and use Markdown headings with #, ##, and ###.\n"
        "Include an introduction, organized body sections, and a conclusion when appropriate.\n"
        "Do not invent citations, statistics, quotations, or sources.\n\n"
    )
    combined_prompt = system_instruction + final_prompt

    model_name = GEMINI_MODEL

    try:
        logger.info("Calling Gemini model=%s", model_name)

        if hasattr(genai, "Client"):
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=model_name,
                contents=combined_prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                },
            )
            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                raise RuntimeError("Empty Gemini response")
            return str(text).strip()

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            combined_prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 8192,
            },
            request_options={"timeout": 25},
        )
        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise RuntimeError("Empty Gemini response")
        return str(text).strip()

    except Exception as e:
        msg = str(e)
        if "429" in msg.upper() or "RESOURCE_EXHAUSTED" in msg.upper() or "quota" in msg.lower():
            raise RuntimeError(f"429 Quota exceeded for {model_name}: {msg}") from e
        raise RuntimeError(f"Gemini generation failed for {model_name}: {msg}") from e

@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ok", "message": "Tempaper API is running."}), 200

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    conn = None
    try:
        conn = get_db()
        otp = generate_otp("ver")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO users (username, email, password_hash, is_verified, otp_code, otp_expires_at)
                   VALUES (%s, %s, %s, FALSE, %s, %s) RETURNING id""",
                (username, email, generate_password_hash(password), otp, expires_at)
            )
            user_id = cur.fetchone()[0]
            conn.commit()

        if not send_email(email, "Verify Your Tempaper Account", "<p>Your code is: <strong>%s</strong></p><p>Expires in 15 minutes.</p>" % otp):
            return jsonify({"error": "Verification email could not be sent. Please try again later."}), 502

        return jsonify({"message": "Registration successful. Check your email.", "user_id": user_id}), 201

    except psycopg2.IntegrityError:
        if conn:
            conn.rollback()
        return jsonify({"error": "Username or Email already exists"}), 400
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Registration failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = str(data.get("identifier", "")).strip().lower()
    password = data.get("password", "")

    if not identifier or not password:
        return jsonify({"error": "Missing email/username or password"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, is_verified FROM users WHERE LOWER(email) = %s OR LOWER(username) = %s",
                (identifier, identifier),
            )
            row = cur.fetchone()

        if not row or not check_password_hash(row[3], password):
            return jsonify({"error": "Invalid username/email or password"}), 401

        if not row[4]:
            return jsonify({"error": "Verify email first"}), 403

        with conn.cursor() as cur:
            cur.execute("UPDATE users SET last_login = %s WHERE id = %s", (datetime.utcnow(), row[0]))
            conn.commit()

        return jsonify({
            "message": "Login successful",
            "user_id": row[0],
            "user": {"id": row[0], "username": row[1], "email": row[2]}
        }), 200

    except Exception as e:
        logger.exception("Login error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/verify-email", methods=["POST"])
def verify_email():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()

    if not email or not code:
        return jsonify({"error": "Email and code are required"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id, otp_code, otp_expires_at FROM users WHERE LOWER(email) = %s", (email,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "User not found"}), 404

            user_id, saved_otp, expires_at = row
            if saved_otp != code:
                return jsonify({"error": "Invalid verification code"}), 400
            if expires_at and datetime.utcnow() > expires_at:
                return jsonify({"error": "Verification code has expired"}), 400

            cur.execute("UPDATE users SET is_verified = TRUE, otp_code = NULL, otp_expires_at = NULL WHERE id = %s", (user_id,))
            conn.commit()

        return jsonify({"message": "Account verified successfully!"}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Verify error: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not email:
        return jsonify({"error": "Email is required"}), 400

    conn = None
    try:
        conn = get_db()
        otp = generate_otp("ver")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

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

        if not send_email(email, "Your Verification Code", "<p>Your new verification code is: <strong>%s</strong></p>" % otp):
            return jsonify({"error": "Verification email could not be sent."}), 502

        return jsonify({"message": "Verification code resent successfully"}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Resend verification failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/request-reset", methods=["POST"])
def request_reset():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    conn = None
    try:
        conn = get_db()
        otp = generate_otp("pass")
        expires_at = datetime.utcnow() + timedelta(minutes=15)

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET otp_code = %s, otp_expires_at = %s WHERE LOWER(email) = %s RETURNING id",
                (otp, expires_at, email)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"message": "If the email exists, a code was sent."}), 200
            conn.commit()

        if not send_email(email, "Password Reset Code", "<h2>Tempaper Password Reset</h2><p>Your password reset code is:</p><h1>%s</h1><p>This code expires in 15 minutes.</p>" % otp):
            return jsonify({"error": "The reset code could not be sent. Please try again later."}), 502

        return jsonify({"message": "Reset code sent to your email."}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Request reset failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    new_password = data.get("new_password", "")

    if not email or not code or not new_password:
        return jsonify({"error": "Missing required fields"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT otp_code, otp_expires_at FROM users WHERE LOWER(email) = %s", (email,))
            row = cur.fetchone()
            if not row or row[0] != code:
                return jsonify({"error": "Invalid code or email"}), 400
            if row[1] and datetime.utcnow() > row[1]:
                return jsonify({"error": "Reset code has expired"}), 400

            cur.execute(
                "UPDATE users SET password_hash = %s, otp_code = NULL, otp_expires_at = NULL WHERE LOWER(email) = %s",
                (generate_password_hash(new_password), email),
            )
            conn.commit()

        return jsonify({"message": "Password updated successfully!"}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Reset password failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

# Use caching decorator safely: our cache wrapper provides cached(...) in all fallback cases
@app.route("/me", methods=["GET", "HEAD"])
@cache.cached(timeout=300, query_string=True)
def get_me():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    if not str(user_id).isdigit():
        return jsonify({"user": {"id": 0, "username": "Guest User", "email": "guest@tempaper.com", "is_verified": False}}), 200

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, email, is_verified FROM users WHERE id = %s", (int(user_id),))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "User not found"}), 404

        return jsonify({"user": {"id": row[0], "username": row[1], "email": row[2], "is_verified": row[3]}}), 200

    except Exception as e:
        logger.exception("Get profile failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/generate", methods=["POST"])
def generate_paper():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    user_id = data.get("user_id", "guest")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if len(prompt) > 20000:
        return jsonify({"error": "Prompt is too long"}), 413
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is missing on the server"}), 503

    try:
        tavily_context, sources = search_tavily(prompt)
        final_content = "Topic/Prompt: %s\n\nReference Material:\n%s" % (prompt, tavily_context) if tavily_context else prompt

        result_text = generate_with_gemini(prompt, final_content)
        word_count = len(result_text.split())
        page_count = max(1, word_count // 250)

        # Only save sessions for real numeric users (guests do not get saved)
        if str(user_id).isdigit():
            conn = None
            try:
                conn = get_db()
                session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO paper_sessions (user_id, session_id, title, prompt_text, result_text, word_count, page_count)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (int(user_id), session_id, prompt[:200], prompt, result_text, word_count, page_count)
                    )
                    conn.commit()
            except Exception as e:
                logger.warning("Could not save paper session: %s", e)
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    release_db(conn)

        return jsonify({"result": result_text, "sources": sources, "word_count": word_count, "page_count": page_count}), 200

    except RuntimeError as e:
        logger.exception("Generation failed: %s", e)
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return jsonify({"error": "AI quota exceeded. Free tier is 5 req/min. Please wait 1 min and retry."}), 429
        return jsonify({"error": "Gemini is temporarily unavailable. Please try again in a moment.", "details": msg[:400]}), 503
    except Exception as e:
        logger.exception("Generate route failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/papers", methods=["GET"])
def get_papers():
    user_id = request.args.get("user_id")
    # Guests and invalid user ids get an empty list (no history)
    if not user_id or not str(user_id).isdigit():
        return jsonify({"papers": []}), 200

    # Verify the user actually exists; if not, return empty
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE id = %s", (int(user_id),))
            user_row = cur.fetchone()
            if not user_row:
                return jsonify({"papers": []}), 200

            # Fetch paper sessions strictly for this user
            cur.execute("""
                SELECT id, session_id, title, prompt_text, word_count, page_count, created_at
                FROM paper_sessions WHERE user_id = %s ORDER BY created_at DESC LIMIT 50
            """, (int(user_id),))
            rows = cur.fetchall()

        papers = []
        for row in rows:
            papers.append({
                "id": row[0],
                "session_id": row[1],
                "title": row[2],
                "prompt": row[3],
                "word_count": row[4],
                "page_count": row[5],
                "created_at": row[6].isoformat() if row[6] else None
            })

        return jsonify({"papers": papers}), 200
    except Exception as e:
        logger.exception("Get paper list failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    user_id = request.args.get("user_id")
    if not user_id or not str(user_id).isdigit():
        return jsonify({"error": "Valid user_id required"}), 400

    if request.method == "GET":
        conn = None
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT theme, font_size, notifications, auto_save FROM user_settings WHERE user_id = %s", (int(user_id),))
                row = cur.fetchone()
            if not row:
                return jsonify({"settings": {"theme": "system", "font_size": 16, "notifications": True, "auto_save": True}}), 200
            return jsonify({"settings": {"theme": row[0], "font_size": row[1], "notifications": row[2], "auto_save": row[3]}}), 200
        except Exception as e:
            logger.exception("Get settings failed: %s", e)
            return jsonify({"error": "Internal server error"}), 500
        finally:
            if conn:
                release_db(conn)

    data = request.get_json(silent=True) or {}
    theme = data.get("theme", "system")
    font_size = data.get("font_size", 16)
    notifications = data.get("notifications", True)
    auto_save = data.get("auto_save", True)

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_settings (user_id, theme, font_size, notifications, auto_save)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    theme = EXCLUDED.theme,
                    font_size = EXCLUDED.font_size,
                    notifications = EXCLUDED.notifications,
                    auto_save = EXCLUDED.auto_save,
                    updated_at = CURRENT_TIMESTAMP
            """, (int(user_id), theme, font_size, notifications, auto_save))
            conn.commit()
        return jsonify({"message": "Settings updated successfully"}), 200
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Update settings failed: %s", e)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            release_db(conn)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # app already created above
    app.run(host="0.0.0.0", port=port, debug=False)
