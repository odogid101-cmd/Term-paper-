import os
import random
import string
import logging
import functools
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS

# Cache fallback
cache = None
try:
    from flask_caching import Cache as FlaskCache
    cache = FlaskCache(config={"CACHE_TYPE": "simple", "CACHE_DEFAULT_TIMEOUT": 300})
    app = Flask(__name__)
    cache.init_app(app)
except Exception:
    app = Flask(__name__)
    from cachelib import SimpleCache
    _store = SimpleCache(default_timeout=300)
    class FallbackCache:
        def cached(self, timeout=300, query_string=False):
            def decorator(f):
                @functools.wraps(f)
                def wrapped(*args, **kwargs):
                    key = f"cached:{request.path}"
                    if query_string:
                        key += f"?{request.query_string.decode()}"
                    val = _store.get(key)
                    if val: return val
                    rv = f(*args, **kwargs)
                    _store.set(key, rv, timeout)
                    return rv
                return wrapped
            return decorator
    cache = FallbackCache()

CORS(app, supports_credentials=True, origins=["https://odogid101-cmd.github.io", "http://localhost:3000", "http://localhost:5000"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import psycopg2
from psycopg2 import pool
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"Groq initialized with model {GROQ_MODEL}")
    except Exception as e:
        logger.error(f"Groq init failed: {e}")

db_pool = None
if DATABASE_URL:
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(5, 20, DATABASE_URL)
    except Exception as e:
        logger.exception(f"DB pool error: {e}")

def get_db():
    if not db_pool: raise RuntimeError("DB not initialized")
    return db_pool.getconn()
def release_db(conn):
    if db_pool and conn: db_pool.putconn(conn)

def init_db():
    if not db_pool: return
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username VARCHAR(100) UNIQUE NOT NULL, email VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL, is_verified BOOLEAN DEFAULT FALSE, otp_code VARCHAR(20), otp_expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login TIMESTAMP);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS paper_sessions (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, session_id VARCHAR(64) UNIQUE NOT NULL, title VARCHAR(200), prompt_text TEXT, result_text TEXT, word_count INTEGER DEFAULT 0, page_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS user_settings (id SERIAL PRIMARY KEY, user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE, theme VARCHAR(20) DEFAULT 'system', font_size INTEGER DEFAULT 16, notifications BOOLEAN DEFAULT TRUE, auto_save BOOLEAN DEFAULT TRUE, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
            conn.commit()
    except Exception as e:
        if conn: conn.rollback()
        logger.exception(f"DB init fail: {e}")
    finally:
        if conn: release_db(conn)
init_db()

def generate_otp(prefix): return f"{prefix}{''.join(random.choices(string.digits, k=3))}"

def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY: return False
    try:
        r=requests.post("https://api.resend.com/emails", json={"from":RESEND_FROM_EMAIL,"to":[to_email],"subject":subject,"html":html_content}, headers={"Authorization":f"Bearer {RESEND_API_KEY}"}, timeout=15)
        return r.status_code in (200,201)
    except: return False

def search_tavily(query):
    if not TAVILY_API_KEY: return "", []
    try:
        r=requests.post("https://api.tavily.com/search", json={"api_key":TAVILY_API_KEY,"query":query,"search_depth":"advanced","max_results":5}, timeout=10)
        if r.status_code==200:
            res=r.json().get("results",[])
            snippets=[i.get("content","") for i in res if i.get("content")]
            sources=[i.get("url","") for i in res if i.get("url")]
            return "\n\n".join(snippets), sources
    except Exception as e:
        logger.warning(f"Tavily fail: {e}")
    return "", []

def generate_with_groq(prompt, context_text=""):
    if not groq_client: raise RuntimeError("GROQ_API_KEY missing - get free key at console.groq.com")
    final_prompt = f"{context_text}\n\n{prompt}" if context_text else prompt

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are Tempaper, an expert academic writing assistant. Write a clear, well-structured academic paper based on user's request. Use professional academic tone, include suitable title, Markdown headings #, ##, ###, introduction, organized body sections, conclusion. Do not invent citations, statistics, or sources. Be detailed and thorough. Minimum 800 words."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.7,
        max_tokens=8192,
        top_p=0.9
    )
    text = response.choices[0].message.content
    if not text or not text.strip(): raise RuntimeError("Empty Groq response")
    return text.strip()

# --- ROUTES ---
@app.route("/", methods=["GET","HEAD"])
def health_check(): return jsonify({"status":"ok","message":f"Tempaper API running - Groq {GROQ_MODEL}"}),200

@app.route("/register", methods=["POST"])
def register():
    data=request.get_json(silent=True) or {}
    username=str(data.get("username","")).strip()
    email=str(data.get("email","")).strip().lower()
    password=data.get("password","")
    if not username or not email or not password: return jsonify({"error":"All fields required"}),400
    conn=None
    try:
        conn=get_db()
        otp=generate_otp("ver")
        exp=datetime.utcnow()+timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (username,email,password_hash,is_verified,otp_code,otp_expires_at) VALUES (%s,%s,%s,FALSE,%s,%s) RETURNING id",(username,email,generate_password_hash(password),otp,exp))
            uid=cur.fetchone()[0]
            conn.commit()
        if not send_email(email,"Verify Your Tempaper Account",f"<p>Your code: <strong>{otp}</strong></p>"): return jsonify({"error":"Email could not be sent"}),502
        return jsonify({"message":"Registered","user_id":uid}),201
    except psycopg2.IntegrityError:
        if conn: conn.rollback()
        return jsonify({"error":"Username or Email exists"}),400
    finally:
        if conn: release_db(conn)

@app.route("/login", methods=["POST"])
def login():
    data=request.get_json(silent=True) or {}
    ident=str(data.get("identifier","")).strip().lower()
    pwd=data.get("password","")
    if not ident or not pwd: return jsonify({"error":"Missing fields"}),400
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,username,email,password_hash,is_verified FROM users WHERE LOWER(email)=%s OR LOWER(username)=%s",(ident,ident))
            r=cur.fetchone()
        if not r or not check_password_hash(r[3],pwd): return jsonify({"error":"Invalid credentials"}),401
        if not r[4]: return jsonify({"error":"Verify email first"}),403
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET last_login=%s WHERE id=%s",(datetime.utcnow(),r[0]))
            conn.commit()
        return jsonify({"message":"Login ok","user_id":r[0],"user":{"id":r[0],"username":r[1],"email":r[2]}}),200
    finally:
        if conn: release_db(conn)

@app.route("/verify-email", methods=["POST"])
def verify_email():
    data=request.get_json(silent=True) or {}
    email=str(data.get("email","")).strip().lower()
    code=str(data.get("code","")).strip()
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,otp_code,otp_expires_at FROM users WHERE LOWER(email)=%s",(email,))
            row=cur.fetchone()
            if not row: return jsonify({"error":"User not found"}),404
            if row[1]!=code: return jsonify({"error":"Invalid code"}),400
            if row[2] and datetime.utcnow()>row[2]: return jsonify({"error":"Code expired"}),400
            cur.execute("UPDATE users SET is_verified=TRUE,otp_code=NULL,otp_expires_at=NULL WHERE id=%s",(row[0],))
            conn.commit()
        return jsonify({"message":"Verified"}),200
    finally:
        if conn: release_db(conn)

@app.route("/resend-verification", methods=["POST"])
def resend():
    data=request.get_json(silent=True) or {}
    email=str(data.get("email","")).strip().lower()
    conn=None
    try:
        conn=get_db()
        otp=generate_otp("ver")
        exp=datetime.utcnow()+timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET otp_code=%s,otp_expires_at=%s WHERE LOWER(email)=%s RETURNING is_verified",(otp,exp,email))
            row=cur.fetchone()
            if not row: return jsonify({"error":"Email not found"}),404
            conn.commit()
        send_email(email,"Verification Code",f"<p>Code: <strong>{otp}</strong></p>")
        return jsonify({"message":"Code resent"}),200
    finally:
        if conn: release_db(conn)

@app.route("/request-reset", methods=["POST"])
def req_reset():
    data=request.get_json(silent=True) or {}
    email=str(data.get("email","")).strip().lower()
    conn=None
    try:
        conn=get_db()
        otp=generate_otp("pass")
        exp=datetime.utcnow()+timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET otp_code=%s,otp_expires_at=%s WHERE LOWER(email)=%s RETURNING id",(otp,exp,email))
            if not cur.fetchone(): return jsonify({"message":"If email exists, code sent"}),200
            conn.commit()
        send_email(email,"Reset Code",f"<h1>{otp}</h1>")
        return jsonify({"message":"Code sent"}),200
    finally:
        if conn: release_db(conn)

@app.route("/reset-password", methods=["POST"])
def reset_pwd():
    data=request.get_json(silent=True) or {}
    email=str(data.get("email","")).strip().lower()
    code=str(data.get("code","")).strip()
    np=data.get("new_password","")
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT otp_code,otp_expires_at FROM users WHERE LOWER(email)=%s",(email,))
            r=cur.fetchone()
            if not r or r[0]!=code: return jsonify({"error":"Invalid code"}),400
            if r[1] and datetime.utcnow()>r[1]: return jsonify({"error":"Expired"}),400
            cur.execute("UPDATE users SET password_hash=%s,otp_code=NULL WHERE LOWER(email)=%s",(generate_password_hash(np),email))
            conn.commit()
        return jsonify({"message":"Password updated"}),200
    finally:
        if conn: release_db(conn)

@app.route("/me", methods=["GET"])
@cache.cached(timeout=300, query_string=True)
def get_me():
    uid=request.args.get("user_id")
    if not uid or not str(uid).isdigit(): return jsonify({"user":{"id":0,"username":"Guest","email":"guest@tempaper.com"}}),200
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,username,email,is_verified FROM users WHERE id=%s",(int(uid),))
            r=cur.fetchone()
            if not r: return jsonify({"error":"Not found"}),404
        return jsonify({"user":{"id":r[0],"username":r[1],"email":r[2],"is_verified":r[3]}}),200
    finally:
        if conn: release_db(conn)

@app.route("/generate", methods=["POST"])
def generate_paper():
    data=request.get_json(silent=True) or {}
    prompt=str(data.get("prompt","")).strip()
    user_id=data.get("user_id","guest")
    if not prompt: return jsonify({"error":"Prompt required"}),400
    if len(prompt)>20000: return jsonify({"error":"Prompt too long"}),413
    if not GROQ_API_KEY: return jsonify({"error":"GROQ_API_KEY missing"}),503
    try:
        tavily_context,sources=search_tavily(prompt)
        final_content=f"Topic: {prompt}\n\nReference Material:\n{tavily_context}" if tavily_context else prompt
        result_text=generate_with_groq(prompt, final_content)
        word_count=len(result_text.split())
        page_count=max(1,word_count//250)
        if str(user_id).isdigit():
            conn=None
            try:
                conn=get_db()
                sid=''.join(random.choices(string.ascii_letters+string.digits,k=32))
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO paper_sessions (user_id,session_id,title,prompt_text,result_text,word_count,page_count) VALUES (%s,%s,%s,%s,%s,%s,%s)",(int(user_id),sid,prompt[:200],prompt,result_text,word_count,page_count))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Save fail: {e}")
            finally:
                if conn: release_db(conn)
        return jsonify({"result":result_text,"sources":sources,"word_count":word_count,"page_count":page_count}),200
    except Exception as e:
        logger.exception(f"Generate fail: {e}")
        msg=str(e)
        if "429" in msg: return jsonify({"error":"Groq rate limit, wait 10s"}),429
        return jsonify({"error":"Generation failed","details":msg[:300]}),500

@app.route("/papers", methods=["GET"])
def get_papers():
    uid=request.args.get("user_id")
    if not uid or not str(uid).isdigit(): return jsonify({"papers":[]}),200
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,session_id,title,prompt_text,word_count,page_count,created_at FROM paper_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",(int(uid),))
            rows=cur.fetchall()
        papers=[{"id":r[0],"session_id":r[1],"title":r[2],"prompt":r[3],"word_count":r[4],"page_count":r[5],"created_at":r[6].isoformat() if r[6] else None} for r in rows]
        return jsonify({"papers":papers}),200
    finally:
        if conn: release_db(conn)

@app.route("/settings", methods=["GET","POST"])
def settings():
    uid=request.args.get("user_id") or request.get_json(silent=True).get("user_id")
    if not uid or not str(uid).isdigit(): return jsonify({"error":"Valid user_id required"}),400
    if request.method=="GET":
        conn=None
        try:
            conn=get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT theme,font_size,notifications,auto_save FROM user_settings WHERE user_id=%s",(int(uid),))
                r=cur.fetchone()
            if not r: return jsonify({"settings":{"theme":"system","font_size":16,"notifications":True,"auto_save":True}}),200
            return jsonify({"settings":{"theme":r[0],"font_size":r[1],"notifications":r[2],"auto_save":r[3]}}),200
        finally:
            if conn: release_db(conn)
    data=request.get_json(silent=True) or {}
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_settings (user_id,theme,font_size,notifications,auto_save) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO UPDATE SET theme=EXCLUDED.theme,font_size=EXCLUDED.font_size,notifications=EXCLUDED.notifications,auto_save=EXCLUDED.auto_save,updated_at=CURRENT_TIMESTAMP",(int(uid),data.get("theme","system"),data.get("font_size",16),data.get("notifications",True),data.get("auto_save",True)))
            conn.commit()
        return jsonify({"message":"Updated"}),200
    finally:
        if conn: release_db(conn)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
