from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import os
import random
import datetime
import requests

app = Flask(__name__)
CORS(app)

# =========================
# CONFIG (SET IN RENDER ENV)
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")

# =========================
# DB
# =========================
def get_db():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def init_db():
    conn = get_db()
    cur = conn.cursor()

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
        user_id INTEGER,
        code TEXT,
        purpose TEXT,
        expires_at TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# UTIL
# =========================
def generate_code(prefix):
    return f"{prefix}{random.randint(100, 999)}"

def send_email(to_email, code):
    requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "from": FROM_EMAIL,
            "to": to_email,
            "subject": "Your Code",
            "html": f"<h2>{code}</h2>"
        }
    )

# =========================
# REGISTER
# =========================
@app.route('/register', methods=['POST'])
def register():
    data = request.json

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("""
        INSERT INTO users (username, email, password_hash)
        VALUES (%s, %s, %s) RETURNING id
        """, (
            data['username'],
            data['email'],
            generate_password_hash(data['password'])
        ))

        user_id = cur.fetchone()[0]
        conn.commit()
    except:
        return jsonify({"error": "User exists"}), 400

    code = generate_code("ver")

    cur.execute("""
    INSERT INTO email_verifications (user_id, code, purpose, expires_at)
    VALUES (%s, %s, %s, %s)
    """, (
        user_id,
        code,
        "signup",
        datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
    ))

    conn.commit()
    conn.close()

    send_email(data['email'], code)

    return jsonify({"message": "Verification code sent"})

# =========================
# LOGIN
# =========================
@app.route('/login', methods=['POST'])
def login():
    data = request.json

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT password_hash, is_verified
    FROM users WHERE email=%s OR username=%s
    """, (data['identifier'], data['identifier']))

    user = cur.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not user[1]:
        return jsonify({"error": "Verify email first"}), 403

    if not check_password_hash(user[0], data['password']):
        return jsonify({"error": "Wrong password"}), 401

    return jsonify({"message": "Login successful"})

# =========================
# REQUEST RESET
# =========================
@app.route('/request-reset', methods=['POST'])
def request_reset():
    data = request.json

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email=%s", (data['email'],))
    user = cur.fetchone()

    if not user:
        return jsonify({"error": "User not found"}), 404

    code = generate_code("pass")

    cur.execute("""
    INSERT INTO email_verifications (user_id, code, purpose, expires_at)
    VALUES (%s, %s, %s, %s)
    """, (
        user[0],
        code,
        "reset_password",
        datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
    ))

    conn.commit()
    conn.close()

    send_email(data['email'], code)

    return jsonify({"message": "Reset code sent"})

# =========================
# RESET PASSWORD
# =========================
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    UPDATE users SET password_hash=%s WHERE email=%s
    """, (
        generate_password_hash(data['new_password']),
        data['email']
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "Password updated"})

@app.route('/')
def home():
    return "API running"

if __name__ == "__main__":
    app.run()