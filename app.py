import os
import json
import re
import random
import string
import logging
import functools
import difflib
from datetime import datetime, timedelta
from html.parser import HTMLParser
from html import escape as html_escape, unescape as html_unescape

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
                    if val:
                        return val
                    rv = f(*args, **kwargs)
                    _store.set(key, rv, timeout)
                    return rv
                return wrapped
            return decorator

    cache = FallbackCache()

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": [
        "https://odogid101-cmd.github.io",
        "http://localhost:3000",
        "http://localhost:5000",
        "https://localhost:3000",
    ]}},
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    expose_headers=["Content-Type"],
)

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
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY")

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
    if not db_pool:
        raise RuntimeError("DB not initialized")
    return db_pool.getconn()


def release_db(conn):
    if db_pool and conn:
        db_pool.putconn(conn)


def init_db():
    if not db_pool:
        return
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_verified BOOLEAN DEFAULT FALSE,
                    otp_code VARCHAR(20),
                    otp_expires_at TIMESTAMP,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    session_id VARCHAR(64) UNIQUE NOT NULL,
                    title VARCHAR(200),
                    prompt_text TEXT,
                    result_text TEXT,
                    word_count INTEGER,
                    page_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    theme VARCHAR(20) DEFAULT 'system',
                    font_size INTEGER DEFAULT 16,
                    notifications BOOLEAN DEFAULT TRUE,
                    auto_save BOOLEAN DEFAULT TRUE
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    session_id VARCHAR(64) UNIQUE NOT NULL,
                    title VARCHAR(200),
                    prompt_text TEXT,
                    result_text TEXT,
                    word_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # migration-safe: add last_login if it doesn't already exist
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'last_login'
                """
            )
            if not cur.fetchone():
                cur.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP")

            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception(f"DB init fail: {e}")
    finally:
        if conn:
            release_db(conn)


init_db()


# --- Prompt Builder / staged paper workflow tables ---
def init_workflow_db():
    if not db_pool:
        return
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS paper_prompts (
                id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prompt_code VARCHAR(5) NOT NULL UNIQUE, title VARCHAR(300) NOT NULL,
                config_json JSONB NOT NULL, prompt_text TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS paper_tocs (
                id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prompt_code VARCHAR(5) NOT NULL UNIQUE REFERENCES paper_prompts(prompt_code) ON DELETE CASCADE,
                toc_code VARCHAR(5) NOT NULL UNIQUE, toc_text TEXT NOT NULL,
                sections_json JSONB NOT NULL DEFAULT '[]'::jsonb, page_map_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS paper_projects (
                id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prompt_code VARCHAR(5) NOT NULL REFERENCES paper_prompts(prompt_code) ON DELETE CASCADE,
                toc_code VARCHAR(5) NOT NULL REFERENCES paper_tocs(toc_code) ON DELETE CASCADE,
                current_index INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id,prompt_code,toc_code))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS paper_sections (
                id SERIAL PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES paper_projects(id) ON DELETE CASCADE,
                order_no INTEGER NOT NULL, section_key VARCHAR(100) NOT NULL, section_title VARCHAR(300) NOT NULL,
                content_text TEXT NOT NULL, word_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id,order_no))""")
            # Backward-compatible migrations for merge and final-document editing.
            cur.execute("ALTER TABLE paper_projects ADD COLUMN IF NOT EXISTS merged BOOLEAN NOT NULL DEFAULT FALSE")
            cur.execute("ALTER TABLE paper_projects ADD COLUMN IF NOT EXISTS merged_content TEXT")
        conn.commit()
    except Exception:
        if conn: conn.rollback()
        logger.exception("Workflow DB initialization failed")
    finally:
        if conn: release_db(conn)

init_workflow_db()



RICH_HTML_MARKER = "[[TEMPAPER_RICH_HTML]]"
_ALLOWED_RICH_TAGS = {
    "p", "div", "span", "h1", "h2", "h3", "h4", "strong", "b", "em", "i",
    "u", "s", "del", "ul", "ol", "li", "blockquote", "br", "sub", "sup", "font",
    "img", "figure", "figcaption"
}
# Extra attrs kept for Monkey Scout figures (move/resize/caption persistence)
_FIGURE_DATA_ATTRS = {"data-fig-id", "data-fig-label", "data-fig-url", "data-fig-caption", "data-bound"}
_VOID_RICH_TAGS = {"br", "img"}
_ALLOWED_STYLE_PROPS = {
    "font-family", "font-size", "color", "background-color", "text-align",
    "margin-left", "margin-right", "margin-top", "margin-bottom", "text-indent",
    "width", "max-width", "height", "max-height", "object-fit", "display",
    "float", "clear"
}

class _RichHTMLSanitizer(HTMLParser):
    """Small allow-list sanitizer for editor HTML saved by Tempaper."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in _ALLOWED_RICH_TAGS:
            return
        safe_attrs = []
        legacy_style = []
        for key, value in attrs:
            key = key.lower()
            if not value:
                continue
            if tag == "font" and key in {"face", "color", "size"}:
                if key == "face" and re.fullmatch(r"[A-Za-z0-9 ,_-]{1,80}", value):
                    legacy_style.append("font-family: " + value)
                elif key == "color" and re.fullmatch(r"(#[0-9a-fA-F]{3,8}|[A-Za-z]{3,20})", value):
                    legacy_style.append("color: " + value)
                elif key == "size" and re.fullmatch(r"[1-7]", value):
                    legacy_style.append("font-size: " + str(8 + int(value) * 2) + "pt")
                continue
            # Allow safe attributes for images and figures
                        # Allow safe attributes for images and figures
            if tag == "img" and key in {"src", "alt", "class", "loading", "draggable", "width", "height"}:
                if key == "src" and re.match(r"^https?://[A-Za-z0-9._~:/?#\[\]@!$&\'()*+,;=%-]+$", value):
                    safe_attrs.append((key, value))
                elif key == "alt" and len(value) <= 300:
                    safe_attrs.append((key, value[:300]))
                elif key == "class" and re.fullmatch(r"[A-Za-z0-9 _\-]{1,80}", value):
                    safe_attrs.append((key, value))
                elif key == "loading" and value in {"lazy", "eager"}:
                    safe_attrs.append((key, value))
                elif key == "draggable" and value in {"true", "false"}:
                    safe_attrs.append((key, value))
                elif key in {"width", "height"} and re.fullmatch(r"\d{1,4}%?", value):
                    safe_attrs.append((key, value))
                continue
            if tag in {"div", "figure", "span", "figcaption"} and (
                key in {"class", "data-fig-label", "data-fig-url", "data-fig-caption", "title"} or key.startswith("data-")
            ):
                if key == "class" and re.fullmatch(r"[A-Za-z0-9 _\-]{1,120}", value):
                    safe_attrs.append((key, value))
                elif key == "title" and len(value) <= 200:
                    safe_attrs.append((key, value[:200]))
                elif key.startswith("data-") and len(value) <= 800:
                    # Keep figure metadata (label, imgbb url, caption)
                    safe_attrs.append((key, value[:800]))
                continue
            if key != "style":
                continue
            styles = []
            for declaration in value.split(";"):
                if ":" not in declaration:
                    continue
                prop, val = declaration.split(":", 1)
                prop, val = prop.strip().lower(), val.strip()
                if prop not in _ALLOWED_STYLE_PROPS:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9#(),.%\s'\\\"_-]{1,100}", val):
                    continue
                styles.append(f"{prop}: {val}")
            if styles:
                safe_attrs.append(("style", "; ".join(styles)))
        if legacy_style:
            safe_attrs.append(("style", "; ".join(legacy_style)))
        attr_text = "".join(f' {k}="{html_escape(v, quote=True)}"' for k, v in safe_attrs)
        self.parts.append(f"<{tag}{attr_text}>")
        if tag not in _VOID_RICH_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag not in _ALLOWED_RICH_TAGS or tag in _VOID_RICH_TAGS:
            return
        if tag in self.stack:
            while self.stack:
                current = self.stack.pop()
                self.parts.append(f"</{current}>")
                if current == tag:
                    break

    def handle_data(self, data):
        self.parts.append(html_escape(data))

    def get_html(self):
        while self.stack:
            self.parts.append(f"</{self.stack.pop()}>")
        return "".join(self.parts)

def sanitize_rich_html(value):
    parser = _RichHTMLSanitizer()
    parser.feed(str(value or ""))
    parser.close()
    return RICH_HTML_MARKER + parser.get_html()

class _PlainTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def handle_starttag(self, tag, attrs):
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "blockquote", "figure", "figcaption"}:
            self.parts.append("\n")
        if tag == "img":
            # Prefer alt text so word counts stay meaningful
            for k, v in attrs:
                if k.lower() == "alt" and v:
                    self.parts.append(v + " ")
                    break
    def get_text(self):
        return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", "".join(self.parts))).strip()

def content_plain_text(value):
    value = str(value or "")
    if value.startswith(RICH_HTML_MARKER):
        parser = _PlainTextExtractor()
        parser.feed(value[len(RICH_HTML_MARKER):])
        parser.close()
        return parser.get_text()
    return value


def _inline_markdown_html(text):
    escaped = html_escape(str(text), quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"~~(.+?)~~", r"<del>\1</del>", escaped)
    escaped = re.sub(r"\+\+(.+?)\+\+", r"<u>\1</u>", escaped)
    return escaped

def content_rich_html(value):
    value = str(value or "")
    if value.startswith(RICH_HTML_MARKER):
        return value[len(RICH_HTML_MARKER):]
    output=[]
    paragraph=[]
    list_tag=None
    list_items=[]
    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            output.append("<p>" + _inline_markdown_html(" ".join(paragraph).strip()) + "</p>")
            paragraph=[]
    def flush_list():
        nonlocal list_tag,list_items
        if list_items:
            output.append(f"<{list_tag}>" + "".join("<li>"+_inline_markdown_html(item)+"</li>" for item in list_items) + f"</{list_tag}>")
        list_tag=None
        list_items=[]
    for raw_line in value.replace("\r","").split("\n"):
        line=raw_line.strip()
        if not line:
            flush_paragraph(); flush_list(); continue
        hm=re.match(r"^(#{1,3})\s+(.+)$",line)
        if hm:
            flush_paragraph(); flush_list()
            level=len(hm.group(1))
            output.append(f"<h{level}>"+_inline_markdown_html(hm.group(2))+f"</h{level}>")
            continue
        hm=re.match(r"^\*\*\s*(.+?)\s*\*\*$",line)
        if hm:
            flush_paragraph(); flush_list()
            output.append("<h3>"+_inline_markdown_html(hm.group(1))+"</h3>")
            continue
        om=re.match(r"^\d+[.)]\s+(.+)$",line)
        um=re.match(r"^[-*+]\s+(.+)$",line)
        if om or um:
            flush_paragraph()
            desired="ol" if om else "ul"
            if list_tag and list_tag!=desired: flush_list()
            list_tag=desired
            list_items.append((om or um).group(1))
            continue
        flush_list()
        paragraph.append(line.lstrip("> ").strip())
    flush_paragraph(); flush_list()
    return "".join(output)


def _new_short_code(conn, table, column):
    allowed = {"paper_prompts": "prompt_code", "paper_tocs": "toc_code"}
    if allowed.get(table) != column:
        raise ValueError("Invalid code target")
    with conn.cursor() as cur:
        for _ in range(100):
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
            cur.execute("SELECT EXISTS(SELECT 1 FROM paper_prompts WHERE prompt_code=%s) OR EXISTS(SELECT 1 FROM paper_tocs WHERE toc_code=%s)", (code, code))
            if not cur.fetchone()[0]: return code
    raise RuntimeError("Could not allocate unique code")


def _owned_prompt(conn, user_id, code):
    with conn.cursor() as cur:
        cur.execute("SELECT prompt_code,title,config_json,prompt_text FROM paper_prompts WHERE user_id=%s AND prompt_code=%s",
                    (int(user_id), code.upper()))
        return cur.fetchone()


def _owned_toc(conn, user_id, code):
    with conn.cursor() as cur:
        cur.execute("SELECT prompt_code,toc_code,toc_text,sections_json,page_map_json FROM paper_tocs WHERE user_id=%s AND toc_code=%s",
                    (int(user_id), code.upper()))
        return cur.fetchone()


def _chapter_page_budget(config, title, sections):
    """Allocate the requested main-body page count across chapters only.

    Preliminary pages, references, and appendices are intentionally excluded.
    The word target is an estimate because actual page breaks depend on rendered layout.
    """
    try:
        target_pages = max(1, min(500, int((config or {}).get("pages", 20))))
        requested_chapters = max(1, min(12, int((config or {}).get("chapters", 5))))
    except (TypeError, ValueError):
        target_pages, requested_chapters = 20, 5

    chapter_titles = []
    for item in (sections or []):
        label = str(item.get("title") or "").strip()
        if re.search(r"\bchapter\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", label, re.I):
            if not any(label.lower() == old.lower() for old in chapter_titles):
                chapter_titles.append(label)

    if not chapter_titles:
        chapter_titles = [f"Chapter {n}" for n in range(1, requested_chapters + 1)]

    index = next((i for i, label in enumerate(chapter_titles) if label.lower() == str(title or "").strip().lower()), None)
    if index is None:
        # Some TOCs use different capitalization or a suffix after the chapter number.
        current_match = re.search(r"\bchapter\s+(iv|v|vi|vii|viii|ix|x|xi|xii|[1-9]|1[0-2]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", str(title or ""), re.I)
        if current_match:
            token = current_match.group(1).lower()
            number_words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}
            try:
                current_number = int(token) if token.isdigit() else number_words.get(token, {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12}.get(token))
            except ValueError:
                current_number = None
            if current_number and 1 <= current_number <= len(chapter_titles):
                index = current_number - 1
    if index is None:
        return None

    count = len(chapter_titles)
    weights = []
    for i in range(count):
        if count == 1:
            weight = 1.0
        elif i == 0:
            weight = 0.9
        elif i == count - 1:
            weight = 0.8
        elif i == 1:
            weight = 1.25
        else:
            weight = 1.0
        weights.append(weight)
    budget = target_pages * weights[index] / sum(weights)
    return {"pages": max(1, round(budget, 1)), "words": max(300, round(budget * 300)), "chapter_index": index + 1, "chapter_count": count, "total_pages": target_pages}


_TOPIC_STOPWORDS = {"about", "above", "after", "against", "among", "an", "and", "are", "as", "at", "assessment", "based", "between", "by", "case", "comparative", "consideration", "effect", "effects", "for", "from", "impact", "in", "into", "investigation", "is", "of", "on", "study", "the", "through", "to", "towards", "under", "using", "with"}


def _topic_alignment_terms(topic):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", str(topic or "").lower())
    return list(dict.fromkeys(w for w in words if w not in _TOPIC_STOPWORDS))


def _topic_alignment_issues(content, topic):
    """Flag likely subject drift using distinctive words from the saved project topic."""
    terms = _topic_alignment_terms(topic)
    if len(terms) < 2:
        return []
    body = content_plain_text(content).lower()
    # Ignore the chapter heading so it cannot satisfy the topic check by itself.
    body = "\n".join(line for line in body.splitlines() if not re.match(r"^\s*(?:chapter\s+\w+|#{1,3}\s*\d+(?:\.\d+)*)", line, re.I))
    matched = [term for term in terms if re.search(rf"\b{re.escape(term)}(?:s|es|ed|ing)?\b", body)]
    required = min(2, len(terms))
    return [] if len(matched) >= required else [f"Topic continuity check found only {len(matched)} of {required} required distinctive topic terms ({', '.join(terms)})."]


def _chapter_number(label):
    """Return a 1-based chapter number for Roman, word, or Arabic chapter labels."""
    text = str(label or "").strip()
    m = re.search(r"\bchapter\s+(xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i|1[0-2]|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", text, re.I)
    if m:
        token = m.group(1).lower()
    else:
        # Common academic TOC format: 1. Introduction, 2. Literature Review.
        m = re.match(r"^\s*(1[0-2]|[1-9])[.)]?\s+\S", text)
        if not m:
            return None
        return int(m.group(1))
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}
    romans = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12}
    return int(token) if token.isdigit() else words.get(token, romans.get(token))


def _normalize_separate_end_matter_toc(toc_text):
    """Ensure Conclusion/Recommendations are standalone TOC entries, not 5.x subsections."""
    output = []
    moved = []
    for raw in str(toc_text or "").splitlines():
        line = raw.strip()
        bare = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", line).strip()
        if re.match(r"^(conclusion|conclusions)\b", bare, re.I):
            moved.append("Conclusion")
            continue
        if re.match(r"^recommendations?\b", bare, re.I):
            moved.append("Recommendations")
            continue
        output.append(raw)
    # Preserve only one standalone entry, in the agreed order, at the end of main matter.
    if not any(x.lower() == "conclusion" for x in moved): moved.insert(0, "Conclusion")
    for item in ("Conclusion", "Recommendations"):
        if item in moved and not any(re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", x).strip().lower() == item.lower() for x in output):
            output.append(item)
    return "\n".join(output).strip()


def _strip_separate_end_matter_from_final_chapter(content, chapter_title, config):
    """Keep standalone Conclusion/Recommendations out of the final numbered chapter."""
    number = _chapter_number(chapter_title)
    try:
        total = max(1, min(12, int((config or {}).get("chapters", 5))))
    except (TypeError, ValueError):
        total = 5
    if number != total:
        return content
    lines = str(content or "").splitlines()
    output = []
    skipping = False
    for line in lines:
        plain = re.sub(r"^\s*#{1,4}\s*", "", line).strip()
        plain = re.sub(r"^\s*\*\*|\*\*\s*$", "", plain).strip()
        plain = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", plain).strip()
        is_end_heading = bool(re.fullmatch(r"(?:conclusions?|recommendations?|references?|bibliography|works cited|appendices?)", plain, re.I))
        is_any_heading = bool(re.match(r"^\s*(?:#{1,4}\s+|\*\*.+\*\*\s*$|\d+(?:\.\d+)+[.)]?\s+)", line))
        if is_end_heading:
            skipping = True
            continue
        if skipping and is_any_heading:
            skipping = False
        if not skipping:
            output.append(line)
    return "\n".join(output).strip()


def _chapter_toc_block(toc_text, chapter_title):
    """Return only one chapter's TOC heading/subheadings for its generation prompt."""
    target = _chapter_number(chapter_title)
    if target is None:
        wanted = str(chapter_title or "").strip().lower()
        matches = []
        for raw in str(toc_text or "").splitlines():
            bare = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", raw).strip().lower()
            if bare == wanted or (wanted == "conclusion" and bare.startswith("conclusion")) or (wanted == "references" and re.match(r"^(references?|bibliography|works cited)\b", bare)):
                matches.append(raw)
        return "\n".join(matches) or str(chapter_title or "")
    selected = []
    active = False
    for raw in str(toc_text or "").splitlines():
        line = raw.strip()
        number = _chapter_number(line)
        if number is not None:
            if active:
                break
            if number == target:
                active = True
                selected.append(raw)
            continue
        bare = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", line).strip()
        if active and re.match(r"^(conclusion|conclusions|recommendations?|references?|bibliography|works cited|appendices?)\b", bare, re.I):
            break
        if active:
            selected.append(raw)
    return "\n".join(selected).strip() or str(toc_text or "")


def _section_items(toc_text, config):
    """Create one generation unit per chapter plus separately generated end matter.

    Decimal headings (1.1, 1.2, 2.1...) remain inside their parent chapter and are
    supplied to the chapter prompt as its authoritative subsection checklist.
    """
    config = config or {}
    requested = max(1, min(12, int(config.get("chapters", 5) or 5)))
    raw_lines = [re.sub(r"[*#`]", "", line).strip() for line in (toc_text or "").splitlines()]
    raw_lines = [re.sub(r"\s+(?:\.{2,}|…+)\s*[ivxlcdm\d]+\s*$", "", x, flags=re.I).strip() for x in raw_lines]
    chapter_map = {}
    end_items = []
    for line in raw_lines:
        if not line or len(line) > 240 or line.lower() in {"table of contents", "contents"}:
            continue
        number = _chapter_number(line)
        if number and 1 <= number <= 12:
            # Exclude numbered subsections (decimal notation) from chapter recognition.
            if re.match(r"^\s*\d+\.\d+", line):
                continue
            clean = re.sub(r"^\s*(?:chapter\s+(?:xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i|1[0-2]|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*[:.)-]?|(?:1[0-2]|[1-9])[.)]?\s*)", "", line, flags=re.I).strip()
            # Preserve a useful title while making the numbering canonical.
            title = f"Chapter {number}" + (f": {clean}" if clean and not re.fullmatch(r"chapter\s*\d+", clean, re.I) else "")
            chapter_map.setdefault(number, title)
            continue
        bare = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", line).strip()
        low = bare.lower()
        if re.match(r"^(conclusion|conclusions)\b", low):
            if not any(x["title"].lower() == "conclusion" for x in end_items): end_items.append({"key":"conclusion","title":"Conclusion"})
        elif re.match(r"^recommendations?\b", low):
            if not any(x["title"].lower() == "recommendations" for x in end_items): end_items.append({"key":"recommendations","title":"Recommendations"})
        elif re.match(r"^(references?|bibliography|works cited)\b", low):
            if not any(x["title"].lower() == "references" for x in end_items): end_items.append({"key":"references","title":"References"})
        elif re.match(r"^append(?:ix|ices)\b", low):
            if not any(x["title"].lower() == "appendices" for x in end_items): end_items.append({"key":"appendices","title":"Appendices"})

    if chapter_map:
        # Keep only the requested count, but require a contiguous, ordered chapter sequence.
        chapters = []
        for n in range(1, requested + 1):
            chapters.append(chapter_map.get(n, f"Chapter {n}"))
    else:
        chapters = [f"Chapter {n}" for n in range(1, requested + 1)]

    # The user's selected default is a standalone conclusion after the final chapter.
    if not any(x["title"].lower() == "conclusion" for x in end_items):
        end_items.insert(0, {"key":"conclusion","title":"Conclusion"})
    if not any(x["title"].lower() == "references" for x in end_items):
        end_items.append({"key":"references","title":"References"})
    final = [{"key":f"chapter-{i}","title":title} for i,title in enumerate(chapters,1)]
    # Keep conventional end-matter ordering and prevent chapter conclusion becoming 5.6.
    order = {"conclusion":0,"recommendations":1,"references":2,"appendices":3}
    end_items.sort(key=lambda x: order.get(x["key"], 9))
    return final + end_items


@app.route("/prompt-builder", methods=["POST", "GET"])
def prompt_builder():
    if request.method == "GET":
        uid = request.args.get("user_id", "")
        if not uid.isdigit(): return jsonify({"error": "Please sign in to use Prompt Builder."}), 401
        conn = None
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT prompt_code,title,config_json,prompt_text,created_at FROM paper_prompts WHERE user_id=%s ORDER BY created_at DESC LIMIT 50", (int(uid),))
                rows = cur.fetchall()
            return jsonify({"prompts":[{"prompt_code":r[0],"title":r[1],"config":r[2],"prompt_text":r[3],"created_at":r[4].isoformat() if r[4] else None} for r in rows]}), 200
        finally:
            if conn: release_db(conn)
    data=request.get_json(silent=True) or {}
    uid=str(data.get("user_id","")); config=data.get("config") or {}
    title=str(config.get("topic") or data.get("title") or "").strip()
    if not uid.isdigit(): return jsonify({"error":"Sign in to save a prompt to your account."}),401
    if not title: return jsonify({"error":"Enter the term-paper topic."}),400
    try:
        config.update({
            "topic":title, "pages":max(1,min(500,int(config.get("pages",20)))),
            "chapters":max(1,min(12,int(config.get("chapters",5)))),
            "conclusion_mode":"separate",
            "body_font_size":max(8,min(24,int(config.get("body_font_size",12)))),
            "heading1_size":max(8,min(36,int(config.get("heading1_size",14)))),
            "heading2_size":max(8,min(32,int(config.get("heading2_size",14)))),
            "heading3_size":max(8,min(28,int(config.get("heading3_size",12))))
        })
    except (ValueError,TypeError): return jsonify({"error":"Page, chapter, and font-size values must be numbers."}),400
    # Persist cover metadata so the generation prompt can fill known details and leave blanks for unknown ones.
    cover_fields = {
        "institution": "Institution / University", "faculty": "Faculty", "department": "Department",
        "course_code": "Course Code", "course_title": "Course Title", "student_name": "Student Name",
        "registration_number": "Registration / Matriculation Number", "lecturer_name": "Lecturer / Supervisor",
        "academic_session": "Academic Session", "submission_date": "Date of Submission"
    }
    for field in cover_fields:
        config[field] = str(config.get(field) or "").strip()
    cover_lines = "\n".join(f"- {label}: {config[field] if config[field] else '[Leave a blank line for the student to fill in]'}" for field, label in cover_fields.items())
    automatic_rules = '''AUTOMATIC TEMPAPER STRUCTURE RULES — ALWAYS APPLY TO THIS PAPER AND EVERY NEXT PULL:
PAGE-COUNT RULE (IMPORTANT): The requested page count in the prompt builder applies ONLY to the main body, starting at Chapter One and ending at the final chapter. Do NOT count the cover page, certification/approval, declaration, dedication, acknowledgements, abstract, table of contents, lists of tables/figures, references/bibliography, or appendices toward this requested number. Those pages are additional. Across all chapters combined, aim for the requested number of rendered A4 pages. Each chapter prompt will provide its own proportional page and word target; follow that target rather than applying a fixed minimum or an oversized page target to every chapter. As a rough drafting guide, aim for about 280–320 words per body page, adjusted naturally for headings, lists, tables, equations, and topic complexity. Do not claim an exact page count in the prose.
CHAPTER 1 (opening/introduction/background): use the allocated page target; usually 2–3 numbered subheadings and one simple numbered list for objectives/questions; no nested A/B list unless genuinely required.
MIDDLE CHAPTERS: allocate depth according to each chapter's supplied target. A literature/conceptual review can receive a larger share than other chapters when the budget allows. Rotate heading and list structures naturally; do not force irrelevant lists or exceed the total requested main-body page budget.
FINAL NUMBERED CHAPTER: follow its allocated page target and cover the final chapter-specific analysis/findings required by its TOC. Do NOT place a standalone Conclusion or Recommendations inside this chapter. The document-level Conclusion is a separate heading generated immediately after the final numbered chapter; Recommendations, when required, follow that separate Conclusion. Do not force the final chapter to exceed its allocated share.
SUBHEADING STYLE AND COMPLETENESS: use bold numbered subheadings (e.g. 2.1, 2.2) exactly as listed in the saved TOC and in the same order. Each explanatory subheading must contain a developed, topic-specific explanation (normally at least 5–7 substantial displayed lines or equivalent detail, but do not pad with repetition). Classify each subsection by what the topic requires: explanation, list, example, comparison, procedure, calculation/solution, or a combination. Do not force long prose into a subsection whose correct answer is a concise calculation or direct result.
LIST EXPLANATION RULE: when the prose introduces a list as types, factors, causes, reasons, characteristics, elements, functions, advantages/disadvantages, stages, methods, or similar items that require discussion, explain EACH list item after naming it. A label-only list is incomplete when the context requires explanation. Each item should normally include a clear sentence explaining what it means, why it matters, or how it applies, plus an example when useful. Keep list items under the correct parent subheading; never let list items become accidental numbered headings. Not every list needs expanded explanations: contents pages, objectives, short factual answers, and simple enumerations may remain concise when academically appropriate. Finish the current subheading completely before starting the next one.
Simple lists use plain 1, 2, 3 numbering, not bold. Nested lists use bold A. and B. labels with explanations only where a parent point has distinct types; at most one nested A/B list in any chapter. Do not add lists merely to satisfy a count if they make the discussion unnatural; preserve the closest applicable pattern and academic relevance.
COVER PAGE: include the topic in a prominent centered bold heading and all available metadata below. For every missing item, print a clear blank line such as ____________________, never invent personal/institution/course details.
PRELIMINARY PAGES: arrange each page heading centered and bold; body paragraphs justified. Use conventional academic order where appropriate: cover/title page, certification/approval, declaration, dedication, acknowledgements, abstract, table of contents, list of tables/figures/abbreviations if relevant. Cover page has no visible page number; preliminary pages use lower-case Roman numerals; main chapters use Arabic numerals. Keep preliminary headings centered and bold, and body paragraphs justified.
TABLE OF CONTENTS: include preliminary items, every chapter and its subheadings, conclusion/recommendations, references and appendices when relevant. Do not invent page numbers in generated prose. The app will recalculate TOC page numbers from the current rendered/paginated document after each Pull Next or edit/regeneration; preserve TOC entry wording and update matching page values rather than creating a new TOC code.
ACADEMIC QUALITY: Produce university textbook-level depth. Formal clear language, meaningful transitions, topic-specific discussion with thorough explanations, critical analysis, relevant theories and examples, credible sources and consistent citations. Do not fabricate sources, quotations, data, research findings, or claims. Avoid repetition, shallow writing and padding. Each next pull must follow this full rule set plus the saved TOC and current chapter's role.

DOCUMENT QUALITY (ALWAYS):
Write like a professional Word term-paper, not a chatbot.
- Only section body. No AI voice, prefaces, or meta instructions.
- Zero repetition of ideas, sentences, paragraphs, or figure labels.
- Numbered lists under a heading must be 1, 2, 3… in order (never every item as 1.).
- Headings use ## / ### with exact TOC numbers; do not echo the heading in the first body sentence.
- Bold key terms on first use with **term**. Sparse italics only when needed.
- Equations as plain readable lines (K = 1/2 m v^2). Never LaTeX (frac, text, math delimiters, etc.).
- Figure references once, natural: "As shown in Fig. 2.1, …" — never "Fig. 2.1 Fig. 2.1".
- No invented citations, HTML, or code fences.'''
    prompt_text=(f"Term-paper topic: {title}\nTarget main-body length: {config['pages']} A4 pages from Chapter One through the final chapter; preliminary pages and references are additional.\nRequired chapters: {config['chapters']}.\n"
        f"Cover-page details (leave a blank line for any missing value):\n{cover_lines}\n"
        f"Body font: {config['body_font_size']} pt; Heading 1: {config['heading1_size']} pt ({'bold' if config.get('heading1_bold',True) else 'regular'}); "
        f"Heading 2: {config['heading2_size']} pt ({'bold' if config.get('heading2_bold',True) else 'regular'}); "
        f"Heading 3: {config['heading3_size']} pt ({'bold' if config.get('heading3_bold',True) else 'regular'}).\n"
        f"Structure and preliminary pages: {config.get('structure','Flexible academic structure (1–12 chapters)')}.\n"
        f"Chapter count: {config['chapters']} (user-selectable flexible range: 1–12). Conclusion arrangement: separate heading after the final chapter.\n"
        f"Additional instructions: {config.get('instructions','Use formal academic writing, credible sources, and consistent citations.')}\n\n"
        f"{automatic_rules}")
    conn=None
    try:
        conn=get_db(); code=_new_short_code(conn,"paper_prompts","prompt_code")
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_prompts (user_id,prompt_code,title,config_json,prompt_text) VALUES (%s,%s,%s,%s::jsonb,%s)",
                        (int(uid),code,title,json.dumps(config),prompt_text))
        conn.commit()
        return jsonify({"message":"Prompt saved to your account.","prompt_code":code,"title":title,"prompt_text":prompt_text}),201
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("Prompt save failed")
        return jsonify({"error":"Could not save prompt.","details":str(e)[:200]}),500
    finally:
        if conn: release_db(conn)


@app.route("/build-toc", methods=["POST"])
def build_toc():
    data=request.get_json(silent=True) or {}; uid=str(data.get("user_id","")); code=str(data.get("prompt_code","")).strip().upper()
    if not uid.isdigit() or not re.fullmatch(r"[A-Z0-9]{5}",code): return jsonify({"error":"Provide a valid account and five-character prompt code."}),400
    conn=None
    try:
        conn=get_db(); row=_owned_prompt(conn,uid,code)
        if not row: return jsonify({"error":"Prompt code not found in this account."}),404
        with conn.cursor() as cur:
            cur.execute("SELECT toc_code,toc_text FROM paper_tocs WHERE user_id=%s AND prompt_code=%s",(int(uid),code)); existing=cur.fetchone()
        if existing: return jsonify({"message":"Retrieved the saved table of contents.","toc_code":existing[0],"toc":existing[1],"updated":True}),200
        _,title,config,prompt_text=row
        prompt=("Create a complete academic TABLE OF CONTENTS for ONE coherent research project that will be written to university textbook standard. The approved topic is fixed and must govern every chapter; do not split the document into unrelated topics. Include exactly the requested number of chapters, numbered sequentially from Chapter 1 to Chapter N. Each chapter must have a meaningful, topic-specific title and well-organized numbered subsections (1.1, 1.2, 2.1...) that support progressive explanation, analysis and critical discussion. Do not treat subsections as separate chapters. After the final chapter, include a standalone heading exactly 'Conclusion', then Recommendations if appropriate, References, and Appendices if relevant. Conclusion is separate end matter, never numbered as the final chapter's subsection. Include relevant preliminary pages in conventional order. One entry per line; no page numbers; no duplicate headings. Do not write the paper.\n\n"
                f"SAVED PROJECT TOPIC (LOCKED): {title}\nEXACT CHAPTER COUNT: {config.get('chapters',5)}\nSaved instructions:\n{prompt_text}")
        toc_text=normalize_server_text(generate_with_groq(prompt))
        toc_text=_normalize_separate_end_matter_toc(toc_text)
        sections=_section_items(toc_text,config); toc_code=_new_short_code(conn,"paper_tocs","toc_code")
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_tocs (user_id,prompt_code,toc_code,toc_text,sections_json) VALUES (%s,%s,%s,%s,%s::jsonb)",
                        (int(uid),code,toc_code,toc_text,json.dumps(sections)))
        conn.commit()
        return jsonify({"message":"Table of contents created and saved.","toc_code":toc_code,"toc":toc_text,"sections":sections,"updated":False}),201
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("TOC creation failed")
        return jsonify({"error":"Could not create the table of contents.","details":str(e)[:200]}),500
    finally:
        if conn: release_db(conn)


@app.route("/paper-start", methods=["POST"])
def paper_start():
    data=request.get_json(silent=True) or {}; uid=str(data.get("user_id",""))
    pc=str(data.get("prompt_code","")).strip().upper(); tc=str(data.get("toc_code","")).strip().upper()
    if not uid.isdigit() or not re.fullmatch(r"[A-Z0-9]{5}",pc) or not re.fullmatch(r"[A-Z0-9]{5}",tc):
        return jsonify({"error":"Enter both five-character codes and sign in."}),400
    conn=None
    try:
        conn=get_db(); pr=_owned_prompt(conn,uid,pc); toc=_owned_toc(conn,uid,tc)
        if not pr or not toc or toc[0]!=pc: return jsonify({"error":"These codes do not match a saved prompt and TOC in your account."}),404
        with conn.cursor() as cur:
            cur.execute("SELECT id,current_index,merged,merged_content FROM paper_projects WHERE user_id=%s AND prompt_code=%s AND toc_code=%s",(int(uid),pc,tc)); project=cur.fetchone()
        if project:
            with conn.cursor() as cur:
                cur.execute("SELECT order_no,section_title,content_text FROM paper_sections WHERE project_id=%s ORDER BY order_no",(project[0],)); rows=cur.fetchall()
            return jsonify({"message":"Resumed your saved paper.","project_id":project[0],"current_index":project[1],
                            "merged":bool(project[2]),"merged_content":project[3] or "",
                            "title": pr[1] if pr else "",
                            "prompt_code": pc, "toc_code": tc,
                            "sections":[{"order_no":r[0],"title":r[1],"content":r[2]} for r in rows]}),200
        _,title,config,prompt_text=pr; _,_,toc_text,sections,_=toc
        # Rebuild generation units from the saved TOC for new projects, including older
        # TOC codes created before the chapter/subsection parser was hardened.
        toc_text = _normalize_separate_end_matter_toc(toc_text)
        sections = _section_items(toc_text, config)
        with conn.cursor() as cur:
            cur.execute("UPDATE paper_tocs SET toc_text=%s,sections_json=%s::jsonb,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s AND toc_code=%s",
                        (toc_text, json.dumps(sections), int(uid), tc))
        research,sources=search_tavily(prompt_text)
        prompt=("Write ONLY the cover/title page and required preliminary pages to university academic standard. Do NOT generate, include, or preview the Table of Contents. Do NOT write Chapter One or any chapter content. The TOC is a separate stage and will appear only when the user clicks Pull Next after these preliminary pages. Follow every automatic formatting rule in the saved prompt. COVER PAGE: center the full topic in bold; include institution, faculty, department, course title/code, student name, registration number, lecturer/supervisor, academic session, and submission date. For each missing value, leave a visible blank line (____________________); do not invent details. Each preliminary-page heading must be centered and bold; body paragraphs must be justified. Put the cover first with no visible page number; use lower-case Roman numeral pagination for following preliminary pages. Write professional, clear academic language for abstract, acknowledgements etc. Do not include TOC entries or fabricated page numbers.\n\n"
                f"SAVED PROMPT:\n{prompt_text}\n\nResearch notes:\n{research}\n\nWrite the preliminary material only.")
        prelim=remove_duplicate_sections(generate_with_groq(prompt))
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_projects (user_id,prompt_code,toc_code,current_index) VALUES (%s,%s,%s,0) RETURNING id",(int(uid),pc,tc)); pid=cur.fetchone()[0]
            cur.execute("INSERT INTO paper_sections (project_id,order_no,section_key,section_title,content_text,word_count) VALUES (%s,0,'preliminary','Preliminary Pages and Table of Contents',%s,%s)",(pid,prelim,len(prelim.split())))
        conn.commit()
        return jsonify({"message":"Preliminary pages created. Chapter One has not been generated yet.","project_id":pid,"current_index":0,
                        "result":prelim,"toc":toc_text,"toc_code":tc,"title":title,"prompt_code":pc,
                        "sources":sources,"word_count":len(prelim.split()),
                        "page_count":max(1,(len(prelim.split())+249)//250)}),201
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("Paper start failed")
        return jsonify({"error":"Could not start the paper.","details":str(e)[:200]}),500
    finally:
        if conn: release_db(conn)


@app.route("/pull-next", methods=["POST"])
def pull_next():
    data=request.get_json(silent=True) or {}; uid=str(data.get("user_id","")); pid=data.get("project_id")
    action=str(data.get("action","next")).lower()
    if not uid.isdigit() or not str(pid or "").isdigit(): return jsonify({"error":"Sign in and open a saved paper first."}),400
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("""SELECT p.id,p.prompt_code,p.toc_code,p.current_index,pr.prompt_text,pr.config_json,t.toc_text,t.sections_json
                FROM paper_projects p JOIN paper_prompts pr ON pr.prompt_code=p.prompt_code
                JOIN paper_tocs t ON t.toc_code=p.toc_code WHERE p.id=%s AND p.user_id=%s""",(int(pid),int(uid)))
            project=cur.fetchone()
        if not project: return jsonify({"error":"Saved paper not found in your account."}),404
        _,pc,tc,current_index,prompt_text,config,toc_text,sections=project
        sections=sections or []
        if action == "edit_toc":
            incoming=str(data.get("content",""))
            edited_toc=_normalize_separate_end_matter_toc(content_plain_text(incoming).strip())
            if not edited_toc:
                return jsonify({"error":"Table of contents cannot be empty."}),400
            inline_enabled=bool(data.get("inline_enabled",False))
            if inline_enabled:
                refined=generate_with_groq(
                    "Clean up and refine ONLY this academic table of contents. Preserve all chapter and subsection names, their numbering, and their order. "
                    "Use one entry per line. Do not write the paper and do not invent page numbers. Return only the corrected TOC.\n\n"+edited_toc
                )
                edited_toc=normalize_server_text(refined)
            with conn.cursor() as cur:
                cur.execute("UPDATE paper_tocs SET toc_text=%s,sections_json=%s::jsonb,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s AND toc_code=%s",
                            (edited_toc,json.dumps(_section_items(edited_toc,config)),int(uid),tc))
            conn.commit()
            return jsonify({"message":"Table of contents updated.","title":"Table of Contents","result":edited_toc,"inline_applied":inline_enabled,"kind":"toc"}),200

        if action in {"edit", "edit_merged"}:
            incoming=str(data.get("content",""))
            plain_check=content_plain_text(incoming).strip()
            has_figure = bool(re.search(r"<img\b|<figure\b", incoming, re.I))
            if not plain_check and not has_figure:
                return jsonify({"error":"Edited content cannot be empty."}),400

            if action == "edit_merged":
                with conn.cursor() as cur:
                    cur.execute("SELECT merged FROM paper_projects WHERE id=%s AND user_id=%s",(int(pid),int(uid)))
                    merged_row=cur.fetchone()
                    if not merged_row or not merged_row[0]:
                        return jsonify({"error":"This paper has not been merged yet."}),409
                    saved = sanitize_rich_html(incoming[len(RICH_HTML_MARKER):]) if incoming.startswith(RICH_HTML_MARKER) else incoming.strip()
                    cur.execute("UPDATE paper_projects SET merged_content=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s AND user_id=%s",(saved,int(pid),int(uid)))
                conn.commit()
                return jsonify({"message":"Merged document changes saved directly.","title":"Complete Term Paper","result":saved,"merged":True}),200

            edited_plain=content_plain_text(incoming).strip()
            order_no=int(data.get("order_no",current_index))
            with conn.cursor() as cur:
                cur.execute("SELECT merged FROM paper_projects WHERE id=%s AND user_id=%s",(int(pid),int(uid)))
                merged_row=cur.fetchone()
                if not merged_row:
                    return jsonify({"error":"Paper project not found."}),404
                merged=bool(merged_row[0])
            inline_enabled=bool(data.get("inline_enabled",False)) and not merged
            if inline_enabled:
                with conn.cursor() as cur:
                    cur.execute("SELECT section_title FROM paper_sections WHERE project_id=%s AND order_no=%s",(int(pid),order_no))
                    title_row=cur.fetchone()
                    cur.execute("SELECT order_no,section_title,content_text FROM paper_sections WHERE project_id=%s AND order_no<>%s ORDER BY order_no",(int(pid),order_no))
                    neighbors=cur.fetchall()
                if not title_row:
                    return jsonify({"error":"Section not found."}),404
                title=title_row[0]
                previous=next((content_plain_text(r[2]) for r in reversed(neighbors) if r[0] < order_no), "")
                following=next((content_plain_text(r[2]) for r in neighbors if r[0] > order_no), "")
                refine_prompt=(
                    "Refine ONLY the edited academic section below so its paragraphs, subheadings, transitions, and chapter flow smoothly. "
                    "Use the surrounding context only to maintain coherence. Preserve the user's intended meaning, facts, key points, heading numbering, "
                    "and any required equations or citations. Do not invent sources, add unrelated headings, rewrite the entire paper, or mention that you are an AI. "
                    "Return only the improved section text.\n\n"
                    f"Saved term-paper instructions:\n{prompt_text}\n\nSaved TOC:\n{toc_text}\n\n"
                    f"Previous section context:\n{previous[-3500:]}\n\nNext section context:\n{following[:3500]}\n\n"
                    f"Section title: {title}\n\nUser-edited section to refine:\n{edited_plain}"
                )
                saved=remove_duplicate_sections(generate_with_groq(refine_prompt))
            else:
                saved = sanitize_rich_html(incoming[len(RICH_HTML_MARKER):]) if incoming.startswith(RICH_HTML_MARKER) else incoming.strip()
            plain=content_plain_text(saved)
            with conn.cursor() as cur:
                cur.execute("UPDATE paper_sections SET content_text=%s,word_count=%s,updated_at=CURRENT_TIMESTAMP WHERE project_id=%s AND order_no=%s RETURNING section_title",(saved,len(plain.split()),int(pid),order_no)); row=cur.fetchone()
                if not row:return jsonify({"error":"Section not found."}),404
                cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s",(tc,))
                # Keep the merged export synchronized if a project was merged before a direct edit.
                if merged:
                    cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s ORDER BY order_no",(int(pid),))
                    all_rows=cur.fetchall()
                    merged_text="\n\n".join((f"# {r[0]}\n\n{content_plain_text(r[1])}" if r[0] else content_plain_text(r[1])) for r in all_rows)
                    cur.execute("UPDATE paper_projects SET merged_content=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s",(merged_text,int(pid)))
            conn.commit()
            return jsonify({"message":"Section refined with Groq." if inline_enabled else "Section updated and saved directly.",
                            "title":row[0],"result":saved,"order_no":order_no,"inline_applied":inline_enabled,"merged":merged}),200
        if action=="resend":
            order_no=int(data.get("order_no",current_index))
            resend_mode=str(data.get("mode") or data.get("resend_mode") or "full").strip().lower()
            if resend_mode not in {"full", "local"}:
                resend_mode = "full"
            with conn.cursor() as cur:
                cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s AND order_no=%s",(int(pid),order_no)); row=cur.fetchone()
            if not row:return jsonify({"error":"There is no saved section to regenerate."}),404
            title=row[0]
            resend_topic = str(config.get("topic") or "").strip()
            resend_toc_block = _chapter_toc_block(toc_text, title)
            prior_limit = 12 if title.strip().lower() in {"conclusion", "references", "bibliography", "works cited"} else 3
            with conn.cursor() as cur:
                cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s AND order_no>0 AND order_no<%s ORDER BY order_no DESC LIMIT %s", (int(pid), order_no, prior_limit))
                resend_previous = cur.fetchall()
            resend_context = "\n\n".join(f"PREVIOUS SECTION: {r[0]}\n{content_plain_text(r[1])[:1200]}" for r in reversed(resend_previous))
            # Prefer client-sent content (includes figures the user just placed)
            incoming = str(data.get("content") or "").strip()
            source_content = incoming if incoming else (row[1] or "")

            # Extract figure blocks so we can re-attach them after regeneration
            figure_blocks = []
            figure_notes = []
            raw_html = source_content
            if source_content.startswith(RICH_HTML_MARKER):
                raw_html = source_content[len(RICH_HTML_MARKER):]
                        # Pull out each .tp-figure block (balanced div match; nested handles OK)
            def _extract_tp_figures(html):
                out = []
                lower = html.lower()
                start_token = 'class="tp-figure"'
                pos = 0
                while True:
                    i = lower.find(start_token, pos)
                    if i < 0:
                        break
                    open_div = html.rfind("<div", 0, i)
                    if open_div < 0:
                        pos = i + len(start_token)
                        continue
                    depth = 0
                    j = open_div
                    end_pos = None
                    while j < len(html):
                        if html[j:j+4].lower() == "<div" and (j+4 >= len(html) or html[j+4] in " \t\n>/"):
                            depth += 1
                            j += 4
                            continue
                        if html[j:j+6].lower() == "</div>":
                            depth -= 1
                            j += 6
                            if depth == 0:
                                end_pos = j
                                break
                            continue
                        j += 1
                    if end_pos is None:
                        pos = i + len(start_token)
                        continue
                    out.append(html[open_div:end_pos])
                    pos = end_pos
                return out

            figure_blocks = _dedupe_figure_blocks(_extract_tp_figures(raw_html))
            for block in figure_blocks:
                # Keep resized width across Resend via data-fig-width
                wm = re.search(r'data-fig-width="(\d+)"', block)
                if wm:
                    w = wm.group(1)
                    if re.search(r'style="', block, flags=re.I):
                        if re.search(r'width\s*:', block, flags=re.I):
                            block = re.sub(r'width\s*:\s*[^;"]+', f'width: {w}%', block, count=1, flags=re.I)
                        else:
                            block = re.sub(r'style="', f'style="width: {w}%; ', block, count=1, flags=re.I)
                    else:
                        block = block.replace('class="tp-figure"', f'class="tp-figure" style="width: {w}%; max-width: 100%; margin-left: auto; margin-right: auto"', 1)
                figure_blocks.append(block)
                label, caption, block = _figure_meta(block)
                figure_notes.append(f"- {label}" + (f": {caption}" if caption else ""))

            page_budget = _chapter_page_budget(config, title, sections)
            budget_note = (f"\nTHIS CHAPTER'S SHARE OF THE MAIN-BODY PAGE TARGET: approximately {page_budget['pages']} A4 pages and about {page_budget['words']} words. This is chapter {page_budget['chapter_index']} of {page_budget['chapter_count']}; the complete main body target is {page_budget['total_pages']} pages total. Preliminary pages and references are excluded.\n" if page_budget else "")

            if resend_mode == "local":
                # Surgical update: only touch text near figures when caption matches context
                if not figure_blocks:
                    return jsonify({"error": "Add at least one image before using Update near image."}), 400
                existing_plain = content_plain_text(source_content)
                local_prompt = (
                    "You are editing ONE academic chapter. Your job is SURGICAL — do not rewrite the whole chapter.\n"
                    f"Section title: {title}\n\n"
                    "FIGURES THE USER PLACED (do not invent others):\n"
                    + "\n".join(figure_notes) + "\n\n"
                    "RULES:\n"
                    "1. Read each figure caption. Find the heading or paragraph that already discusses that topic.\n"
                    "2. If a matching heading/paragraph exists, add at most 1–2 natural academic sentences that refer to the figure "
                    "by its exact label (e.g. 'As shown in Fig. 4.3, …'). Keep the rest of that subsection unchanged.\n"
                    "3. If NO heading or paragraph relates to the caption, leave that part of the chapter untouched — "
                    "do not invent a forced discussion just to mention the figure.\n"
                    "4. Do not renumber headings, do not move sections, do not expand unrelated subsections, "
                    "and do not remove existing correct content.\n"
                    "5. Return the COMPLETE section text (all headings and paragraphs), with only the allowed local edits.\n"
                    "6. Do NOT output HTML or image tags.\n"
                    "7. The preserved image object is the only place where its label and caption should appear; never repeat the caption as a heading or standalone line. Refer to it naturally in prose only.\n\n"
                    f"Current section text:\n{existing_plain}"
                )
                text_body = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(local_prompt)))
                text_body = _strip_duplicate_figure_caption_lines(text_body, figure_blocks)
                body_html = content_rich_html(text_body)
                # Prefer keeping figures near user placement anchors when possible
                combined_html = _place_figures_in_html(body_html, figure_blocks)
                saved = RICH_HTML_MARKER + combined_html
                msg_ok = "Updated only text near figures whose captions match a heading or paragraph. Unrelated parts were left unchanged."
            else:
                # Full chapter regeneration (original behaviour)
                fig_instruction = ""
                if figure_notes:
                    fig_instruction = (
                        "\n\nTHIS SECTION ALREADY CONTAINS THE FOLLOWING FIGURES (do not invent others):\n"
                        + "\n".join(figure_notes)
                        + "\nFor EACH figure above you MUST:\n"
                        "1. Discuss the idea named in its caption in the most relevant subsection.\n"
                        "2. Refer to the figure by its exact label at least once in a natural academic sentence "
                        "(e.g. 'As illustrated in Fig. 2.1, …' or 'Fig. 2.1 shows …').\n"
                        "3. Put that reference in the same subsection the caption relates to — not only in a summary at the end.\n"
                        "Write 1–2 clear sentences around each reference so the figure is explained in context.\n"
                        "Do NOT output HTML or image tags — only academic text. Figures are re-attached by the system after generation.\n"
                        "The preserved image object owns its label and caption. Do not reproduce either as a heading or standalone caption line; refer to the figure naturally in prose only.\n"
                    )
                prompt=(
                    "DOCUMENT QUALITY CONTRACT (mandatory): Output only formal section text. "
                    "No AI voice. No repetition of any kind. Sequential list numbering (1, 2, 3...). "
                    "Plain readable equations only (never LaTeX). Headings as ## / ### with TOC numbers. "
                    "Bold key terms with **...**. Write like a professional Word document for academic submission.\n\n"
                    f"Regenerate ONLY this existing section titled {title}. Do not repeat other chapters or preliminary pages. "
                    f"LOCKED RESEARCH TOPIC: {resend_topic}. Keep every part of this section directly relevant to this one topic; do not change subjects or drift into another project. "
                    f"Prior generated context for continuity (do not copy):\n{resend_context}\n\n"
                    "Apply every automatic Tempaper rule in the saved instructions, including chapter role, topic-weight page-depth target, "
                    f"varied heading/list pattern, and numbering. Preserve the section's position in the saved TOC.{budget_note}{fig_instruction}\n\n"
                    f"{prompt_text}\n\nTOC block for this section:\n{resend_toc_block}\n\nSection: {title}"
                )
                text_body = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(prompt)))
                if _chapter_number(title) is not None:
                    topic_issues = _topic_alignment_issues(text_body, resend_topic)
                    if topic_issues:
                        retry_prompt = ("SUBJECT-DRIFT REPAIR REQUIRED. Rewrite this one chapter for the locked research topic only. Preserve its saved chapter title and subsection order.\n"
                                        f"LOCKED TOPIC: {resend_topic}\nDetected issues: {'; '.join(topic_issues)}\nOriginal response:\n{text_body}\n\nInstructions:\n{prompt}")
                        text_body = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(retry_prompt)))
                        if _topic_alignment_issues(text_body, resend_topic):
                            return jsonify({"error":"The regenerated chapter failed the approved-topic check. The saved chapter was left unchanged."}),502
                    text_body = _repair_incomplete_chapter(text_body, title, prompt_text, resend_toc_block)[0]
                    if _topic_alignment_issues(text_body, resend_topic):
                        return jsonify({"error":"The regenerated chapter failed the topic check after quality repair. The saved chapter was left unchanged."}),502
                    text_body = _strip_separate_end_matter_from_final_chapter(text_body, title, config)
                text_body = _strip_duplicate_figure_caption_lines(text_body, figure_blocks)
                if figure_blocks:
                    body_html = content_rich_html(text_body)
                    combined_html = _place_figures_in_html(body_html, figure_blocks)
                    saved = RICH_HTML_MARKER + combined_html
                else:
                    saved = text_body
                msg_ok = (
                    "Full chapter rewritten. Figures were kept and referenced where the text matched."
                    if figure_blocks
                    else "Full chapter rewritten. No figures were in this section, so none were re-attached."
                )
            plain = content_plain_text(saved)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE paper_sections SET content_text=%s,word_count=%s,updated_at=CURRENT_TIMESTAMP WHERE project_id=%s AND order_no=%s",
                    (saved, len(plain.split()), int(pid), order_no),
                )
                cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s", (tc,))
            conn.commit()
            return jsonify({
                "message": msg_ok,
                "title": title,
                "result": saved,
                "order_no": order_no,
                "mode": resend_mode,
            }), 200

        next_index=int(current_index)+1
        if next_index>len(sections): return jsonify({"message":"All sections in the saved table of contents have been generated.","done":True}),200
        item=sections[next_index-1]; title=str(item.get("title") or f"Section {next_index}"); key=str(item.get("key") or f"section-{next_index}")
        with conn.cursor() as cur:
            cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s AND order_no=%s",(int(pid),next_index)); existing=cur.fetchone()
        if existing:
            with conn.cursor() as cur:
                cur.execute("UPDATE paper_projects SET current_index=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s",(next_index,int(pid)))
            conn.commit(); return jsonify({"message":"Resumed the next saved section.","title":existing[0],"result":existing[1],"order_no":next_index,"current_index":next_index}),200
        title_low = title.lower()
        chapter_titles = [str(x.get("title") or "") for x in sections if _chapter_number(str(x.get("title") or "")) is not None]
        is_last_chapter = bool(chapter_titles and title.strip().lower() == chapter_titles[-1].strip().lower())
        if any(term in title_low for term in ("references", "bibliography", "appendix", "appendices")):
            chapter_role = "supplementary section"
        elif re.search(r"chapter\s*(1|i|one)\b|introduction|background", title, re.I):
            chapter_role = "opening chapter"
        elif is_last_chapter or re.search(r"conclusion|recommendations?$", title, re.I):
            chapter_role = "closing chapter"
        else:
            chapter_role = "middle chapter"
        # Give the model concrete continuity context from already-saved chapters.
        prior_limit = 12 if title.strip().lower() in {"conclusion", "references", "bibliography", "works cited"} else 3
        with conn.cursor() as cur:
            cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s AND order_no>0 AND order_no<%s ORDER BY order_no DESC LIMIT %s", (int(pid), next_index, prior_limit))
            previous_rows = cur.fetchall()
        context_limit = 1000 if prior_limit > 3 else 1800
        previous_context = "\n\n".join(
            f"PREVIOUS SECTION: {r[0]}\n{content_plain_text(r[1])[:context_limit]}" for r in reversed(previous_rows)
        ) or "No earlier chapter has been generated yet."
        project_topic = str(config.get("topic") or "").strip()
        chapter_toc_text = _chapter_toc_block(toc_text, title)
        page_budget = None if re.match(r"^(conclusion|recommendations?|references?|bibliography|appendices?)$", title.strip(), re.I) else _chapter_page_budget(config, title, sections)
        budget_note = (f"THIS CHAPTER'S SHARE OF THE MAIN-BODY PAGE TARGET: approximately {page_budget['pages']} A4 pages and about {page_budget['words']} words. This is chapter {page_budget['chapter_index']} of {page_budget['chapter_count']}; across all chapters, aim for {page_budget['total_pages']} A4 pages total from Chapter One to the final chapter. Cover and preliminary pages, references, and appendices are additional and do not count toward that total. Aim for roughly 280–320 words per rendered body page; use the supplied word target as a guide, not filler.\n" if page_budget else "This is supplementary material and does not count toward the requested main-body page total.\n")
        prompt=(
            "DOCUMENT QUALITY CONTRACT (mandatory): Output only formal section text. "
            "No AI voice. No repetition of any kind. Sequential list numbering (1, 2, 3...). "
            "Plain readable equations only (never LaTeX). Headings as ## / ### with TOC numbers. "
            "Bold key terms with **...**. Write like a professional Word document for academic submission.\n\n"
            "UNIVERSITY TEXTBOOK STANDARD: Explain concepts thoroughly. Define terms accurately. Provide background, theories, components, applications, limitations and critical analysis where relevant. "
            "Connect ideas progressively. Use examples and evidence. Adapt to the discipline. Avoid shallow or repetitive writing, empty phrases, and artificial padding. "
            "Never invent research findings or data. Generate original explanations.\n\n"
            f"Write ONLY the next section titled exactly: {title}. Do not write any other chapter or repeat content already generated. "
            f"LOCKED RESEARCH TOPIC: {project_topic}. Every paragraph, example, method, finding, and recommendation must directly serve this exact topic. "
            "Do not change the research subject, introduce a second project, or drift into an adjacent topic. Use prior chapter context to maintain continuity, but do not repeat its wording. "
            f"PRIOR GENERATED CONTEXT (continuity reference only):\n{previous_context}\n\n"
            f"Apply ALL automatic Tempaper rules in the saved prompt. Current role: {chapter_role}. {budget_note}"
            "Use the saved TOC as the authoritative heading checklist: include every relevant subsection for this chapter in the exact order and preserve its wording. "
            "Finish each subsection before starting the next. Identify whether each subsection needs explanation, list, example, comparison, calculation/solution, or a combination. "
            "When a list is introduced as types, factors, causes, reasons, characteristics, elements, functions, advantages/disadvantages, stages, or methods that need discussion, explain EACH item under that same subsection; do not leave label-only items. "
            "Do not force extra prose into direct-answer/calculation sections or simple objective lists. "
            "Use numbered bold subheadings, appropriate plain numbered lists and at most one nested A/B list when required by the chapter pattern. "
            "Keep the chapter substantive and informative; do not pad with repetition. Use academic tone, critical reasoning and credible citations from provided research notes. Do not add a new TOC or fabricate page numbers or sources.\n\n"
            f"Original saved instructions (mandatory on every pull):\n{prompt_text}\n\nAuthoritative TOC block for THIS section only:\n{chapter_toc_text}\n\nSection {next_index} of {len(sections)}. Write this section only."
        )
        if re.match(r"^(references?|bibliography|works cited)$", title.strip(), re.I):
            prompt += ("\n\nREFERENCE INTEGRITY RULE: Build the reference list from the verifiable research notes and source URLs supplied with this request, plus sources explicitly cited in the saved chapters. "
                       "Do not invent authors, publication titles, journal names, dates, volume/issue numbers, page ranges, DOIs, or URLs. Use the requested citation style where specified. "
                       "Return real reference entries; never return only a heading or a blank page.")
        research,sources=search_tavily(f"{project_topic} scholarly sources references {title}")
        content=_dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(prompt,research)))
        if _chapter_number(title) is not None:
            topic_issues = _topic_alignment_issues(content, project_topic)
            if topic_issues:
                retry = ("SUBJECT-DRIFT REPAIR REQUIRED. Rewrite this one chapter so it is unmistakably and substantively about the locked project topic. "
                         "Do not switch to another topic. Keep the saved chapter title and TOC subsection order. Return the complete chapter only.\n\n"
                         f"LOCKED TOPIC: {project_topic}\nDetected issue: {'; '.join(topic_issues)}\n"
                         f"Original chapter response:\n{content}\n\nChapter instructions:\n{prompt}")
                content = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(retry, research)))
                remaining_topic_issues = _topic_alignment_issues(content, project_topic)
                if remaining_topic_issues:
                    return jsonify({"error":"The generated chapter did not pass the saved-topic consistency check. Nothing was saved. Please retry generation.", "checks":remaining_topic_issues}),502
        if re.match(r"^(references?|bibliography|works cited)$", title.strip(), re.I):
            # A heading-only response must never be committed as a successful blank references page.
            ref_lines = [ln.strip() for ln in content_plain_text(content).splitlines() if ln.strip()]
            if len(ref_lines) < 2 or len(content_plain_text(content).split()) < 45:
                retry_prompt = prompt + "\n\nCRITICAL: Return actual formatted reference entries, not a heading or blank page. Include only sources genuinely cited/used in this paper; never fabricate authors, titles, journals, DOIs, or URLs. If no verifiable sources are available, state that clearly rather than returning an empty page."
                content = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(retry_prompt, research)))
            ref_lines = [ln.strip() for ln in content_plain_text(content).splitlines() if ln.strip()]
            if len(ref_lines) < 2 or len(content_plain_text(content).split()) < 35:
                return jsonify({"error":"References were not generated with usable entries. Nothing was saved; try Pull Next again after checking the research/source settings."}),502
        quality_notes = []
        if _chapter_number(title) is not None:
            content, quality_issues = _repair_incomplete_chapter(content, title, prompt_text, chapter_toc_text, research)
            if quality_issues:
                logger.info("Chapter quality repair for %s: %s", title, "; ".join(quality_issues))
                quality_notes = quality_issues
            # A repair pass must not reintroduce subject drift.
            post_repair_topic_issues = _topic_alignment_issues(content, project_topic)
            if post_repair_topic_issues:
                return jsonify({"error":"The repaired chapter no longer matches the approved project topic. Nothing was saved.", "checks":post_repair_topic_issues}),502
        content = _strip_separate_end_matter_from_final_chapter(content, title, config)
        if len(content_plain_text(content).split()) < 35:
            return jsonify({"error":f"{title} returned too little usable content. Nothing was saved; please retry."}),502
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_sections (project_id,order_no,section_key,section_title,content_text,word_count) VALUES (%s,%s,%s,%s,%s,%s)",(int(pid),next_index,key,title,content,len(content.split())))
            cur.execute("UPDATE paper_projects SET current_index=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s",(next_index,int(pid)))
            cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s",(tc,))
        conn.commit()
        resp = {"message":f"{title} generated.","title":title,"result":content,"order_no":next_index,"current_index":next_index,
                        "sources":sources,"word_count":len(content.split()),"page_count":max(1,(len(content.split())+249)//250),"toc_code":tc,"toc":toc_text}
        if quality_notes:
            resp["quality_notes"] = quality_notes
        return jsonify(resp),201
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("Pull next failed")
        return jsonify({"error":"Could not generate the next section.","details":str(e)[:200]}),500
    finally:
        if conn: release_db(conn)



@app.route("/merge-paper", methods=["POST"])
def merge_paper():
    data=request.get_json(silent=True) or {}
    uid=str(data.get("user_id","")); pid=data.get("project_id")
    if not uid.isdigit() or not str(pid or "").isdigit():
        return jsonify({"error":"Sign in and open a saved paper first."}),400
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("""SELECT p.current_index,p.merged_content,t.sections_json,t.toc_text
                           FROM paper_projects p JOIN paper_tocs t ON t.toc_code=p.toc_code
                           WHERE p.id=%s AND p.user_id=%s""",(int(pid),int(uid)))
            project=cur.fetchone()
            if not project:
                return jsonify({"error":"Saved paper not found in your account."}),404
            current_index,old_content,sections,toc_text=project
            sections=sections or []
            if int(current_index or 0) < len(sections):
                return jsonify({"error":"Generate every section in the saved TOC before merging the complete paper."}),409
            cur.execute("SELECT section_title,content_text FROM paper_sections WHERE project_id=%s ORDER BY order_no",(int(pid),))
            rows=cur.fetchall()
            if not rows:
                return jsonify({"error":"There is no generated content to merge."}),400
            chunks=[]
            toc_inserted=False
            for row_index,(title,content) in enumerate(rows):
                rich=content_rich_html(content)
                plain=content_plain_text(content)
                is_preliminary = (title or "").lower() in {"preliminary pages and table of contents","preliminary pages"}
                if is_preliminary:
                    chunks.append(rich)
                    if not toc_inserted and toc_text:
                        toc_html="<h2>TABLE OF CONTENTS</h2>"+"".join("<p>"+_inline_markdown_html(line.strip())+"</p>" for line in str(toc_text).splitlines() if line.strip())
                        chunks.append(toc_html)
                        toc_inserted=True
                else:
                    plain_start = re.sub(r"^\s*#{1,6}\s*", "", plain).strip()
                    title_already_present = bool(title and plain_start.lower().startswith(str(title).lower().lstrip()))
                    if not title_already_present and title and _chapter_number(str(title)) is not None and _chapter_number(plain_start) == _chapter_number(str(title)):
                        title_already_present = True
                    if title and not title_already_present:
                        chunks.append("<h1>"+html_escape(str(title))+"</h1>"+rich)
                    else:
                        chunks.append(rich)
            if not toc_inserted and toc_text:
                toc_html="<h2>TABLE OF CONTENTS</h2>"+"".join("<p>"+_inline_markdown_html(line.strip())+"</p>" for line in str(toc_text).splitlines() if line.strip())
                chunks.insert(1 if chunks else 0, toc_html)
            merged_content=RICH_HTML_MARKER+"".join(x for x in chunks if x.strip())
            cur.execute("UPDATE paper_projects SET merged=TRUE,merged_content=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s AND user_id=%s",
                        (merged_content,int(pid),int(uid)))
        conn.commit()
        return jsonify({"message":"Paper merged into one document. Inline Edit is now disabled for this merged file.",
                        "merged":True,"merged_content":merged_content,"title":"Complete Term Paper"}),200
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("Paper merge failed")
        return jsonify({"error":"Could not merge the paper.","details":str(e)[:200]}),500
    finally:
        if conn: release_db(conn)


@app.route("/sync-toc-pages", methods=["POST"])
def sync_toc_pages():
    data=request.get_json(silent=True) or {}; uid=str(data.get("user_id","")); code=str(data.get("toc_code","")).upper(); page_map=data.get("page_map") or {}
    if not uid.isdigit() or not re.fullmatch(r"[A-Z0-9]{5}",code): return jsonify({"error":"Valid account and TOC code required."}),400
    conn=None
    try:
        conn=get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT toc_text FROM paper_tocs WHERE user_id=%s AND toc_code=%s",(int(uid),code))
            saved=cur.fetchone()
            if not saved:return jsonify({"error":"TOC code not found in this account."}),404
            toc_text=saved[0] or ""
            # Keep the existing TOC code but refresh displayed page numbers from the current layout.
            refreshed=[]
            for line in toc_text.splitlines():
                plain=re.sub(r"[*#`]", "", line).strip()
                # Treat a leading Roman numeral on a preliminary item as its page value,
                # not as part of the entry title; the final page number belongs in the right column.
                leading_roman = re.match(r"^([ivxlcdm]+)[.)]\s+(.+)$", plain, re.I)
                if leading_roman and re.search(r"\b(certification|declaration|dedication|acknowledg|abstract|table of contents|list of (?:tables|figures|abbreviations|symbols))\b", leading_roman.group(2), re.I):
                    plain = leading_roman.group(2).strip()
                match=None
                for title, info in page_map.items():
                    if plain.lower().startswith(str(title).lower()):
                        match=info
                        break
                if match is None:
                    line_chapter = _chapter_number(plain)
                    if line_chapter is not None:
                        for title, info in page_map.items():
                            if _chapter_number(str(title)) == line_chapter:
                                match = info
                                break
                if match is not None:
                    page=match.get("page_start","")
                    line=re.sub(r"^\s*[ivxlcdm]+[.)]\s+", "", line, flags=re.I)
                    line=re.sub(r"\s*(?:\.{2,}|…+)\s*[ivxlcdm\d]+\s*$","",line,flags=re.I)
                    line=re.sub(r"\s+[ivxlcdm\d]+\s*$","",line,flags=re.I)
                    line=f"{line.rstrip()}  ........  {page}"
                refreshed.append(line)
            new_toc="\n".join(refreshed)
            cur.execute("UPDATE paper_tocs SET toc_text=%s,page_map_json=%s::jsonb,updated_at=CURRENT_TIMESTAMP WHERE user_id=%s AND toc_code=%s",
                        (new_toc,json.dumps(page_map),int(uid),code))
        conn.commit(); return jsonify({"message":"TOC page numbers updated; TOC code unchanged.","toc_code":code,"toc":new_toc}),200
    finally:
        if conn:release_db(conn)



@app.route("/upload-image", methods=["POST", "OPTIONS"])
def upload_image():
    """Proxy image upload to ImgBB. API key stays on the server (IMGBB_API_KEY)."""
    if request.method == "OPTIONS":
        return ("", 204)
    if not IMGBB_API_KEY:
        logger.error("IMGBB_API_KEY is not set on the server")
        return jsonify({"error": "Image upload is not configured. Set IMGBB_API_KEY on the server."}), 503

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "No image file provided."}), 400

    # Basic validation
    filename = (file.filename or "").lower()
    if not any(filename.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")):
        return jsonify({"error": "Unsupported image type. Use JPG, PNG, GIF or WebP."}), 400

    # Size limit ~8 MB
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 8 * 1024 * 1024:
        return jsonify({"error": "Image must be under 8 MB."}), 400

    try:
        import base64
        raw = file.read()
        b64 = base64.b64encode(raw).decode("ascii")

        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            data={
                "key": IMGBB_API_KEY,
                "image": b64,
                "name": (file.filename or "tempaper-figure")[:80],
            },
            timeout=30,
        )
        payload = resp.json() if resp.content else {}
        if resp.status_code != 200 or not payload.get("success"):
            err = (payload.get("error") or {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("error")
            logger.warning("ImgBB upload failed: %s %s", resp.status_code, err)
            return jsonify({"error": str(err) if err else "ImgBB upload failed."}), 502

        data = payload.get("data") or {}
        url = data.get("url") or data.get("display_url")
        if not url:
            return jsonify({"error": "ImgBB did not return an image URL."}), 502

        return jsonify({
            "url": url,
            "display_url": data.get("display_url") or url,
            "delete_url": data.get("delete_url") or "",
            "width": data.get("width"),
            "height": data.get("height"),
            "size": data.get("size"),
        }), 200
    except requests.Timeout:
        return jsonify({"error": "Image upload timed out. Try a smaller file."}), 504
    except Exception as e:
        logger.exception("upload-image failed")
        return jsonify({"error": "Could not upload image.", "details": str(e)[:200]}), 500


def generate_otp(prefix):
    return f"{prefix}{''.join(random.choices(string.digits, k=3))}"


def send_email(to_email, subject, html_content):
    if not RESEND_API_KEY:
        return False
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            },
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


def search_tavily(query):
    if not TAVILY_API_KEY:
        return "", []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
            },
            timeout=10,
        )
        if r.status_code == 200:
            res = r.json().get("results", [])
            snippets = [i.get("content", "") for i in res if i.get("content")]
            sources = [i.get("url", "") for i in res if i.get("url")]
            return "\n\n".join(snippets), sources
    except Exception as e:
        logger.warning(f"Tavily fail: {e}")
    return "", []



def _figure_meta(block):
    """Extract label, caption, width from a tp-figure HTML block."""
    label_m = re.search(r'data-fig-label="([^"]+)"', block, flags=re.I)
    data_caption_m = re.search(r'data-fig-caption="([^"]*)"', block, flags=re.I)
    cap_m = re.search(r'class="fig-caption"[^>]*>(.*?)</span>\s*</span>', block, flags=re.I|re.S)
    if not cap_m:
        cap_m = re.search(r'class="fig-caption"[^>]*>(.*?)</(?:span|div)>', block, flags=re.I|re.S)
    width_m = re.search(r'data-fig-width="(\d+)"', block, flags=re.I)
    label = html_unescape((label_m.group(1) if label_m else "Figure")).strip()
    caption = ""
    if data_caption_m:
        caption = html_unescape(data_caption_m.group(1)).strip()
    elif cap_m:
        caption = re.sub(r"<[^>]+>", "", cap_m.group(1)).strip()
    caption = re.sub(r"^Fig\.?\s*[\d.]+\s*:?\s*", "", caption, flags=re.I).strip()
    # Ensure width style from data-fig-width
    if width_m:
        w = width_m.group(1)
        if re.search(r'style="', block, flags=re.I):
            if re.search(r'width\s*:', block, flags=re.I):
                block = re.sub(r'width\s*:\s*[^;"]+', f'width: {w}%', block, count=1, flags=re.I)
            else:
                block = re.sub(r'style="', f'style="width: {w}%; ', block, count=1, flags=re.I)
        else:
            block = block.replace('class="tp-figure"', f'class="tp-figure" style="width: {w}%; max-width: 100%; margin-left: auto; margin-right: auto"', 1)
    return label, caption, block


def _dedupe_figure_blocks(figure_blocks):
    """Keep one copy of each figure, preferring its first saved occurrence."""
    unique, seen = [], set()
    for block in figure_blocks or []:
        label, caption, normalized = _figure_meta(block)
        id_match = re.search(r'data-fig-id="([^"]+)"', block, flags=re.I)
        key = ("id:" + id_match.group(1).strip().casefold()) if id_match else ("label:" + re.sub(r"\s+", " ", label).strip().casefold())
        if not key or key == "label:":
            key = "caption:" + re.sub(r"\s+", " ", caption).strip().casefold()
        if key and key in seen:
            continue
        if key: seen.add(key)
        unique.append(normalized)
    return unique


def _strip_duplicate_figure_caption_lines(text, figure_blocks):
    """Remove AI-emitted standalone caption lines; the preserved figure owns its caption."""
    labels = []
    for block in figure_blocks or []:
        label, caption, _ = _figure_meta(block)
        labels.append((label, caption))
    kept = []
    for line in str(text or "").splitlines():
        plain = re.sub(r"[*#`]", "", line).strip()
        duplicate = False
        for label, caption in labels:
            if not plain: continue
            norm = re.sub(r"\s+", " ", plain).strip().casefold()
            lab = re.sub(r"\s+", " ", label).strip().casefold()
            cap = re.sub(r"\s+", " ", caption).strip().casefold()
            if norm in {lab, f"{lab}: {cap}".strip(), f"{lab} {cap}".strip()}:
                duplicate = True; break
        if not duplicate: kept.append(line)
    return "\n".join(kept).strip()


def _place_figures_in_html(body_html, figure_blocks):
    """Insert each figure after the paragraph that first refers to it (by label or caption keywords).

    Falls back to appending at the end only when no sensible anchor is found.
    User can still move/resize/delete afterward.
    """
    if not figure_blocks:
        return body_html
    html = body_html or ""
    placed = []
    remaining = []

    # Split into coarse blocks ending with closing tags that form natural insert points
    # Work on the full HTML string with search for reference patterns.
    for raw_block in figure_blocks:
        label, caption, block = _figure_meta(raw_block)
        # Build search patterns for this figure
        patterns = []
        # Fig. 2.1 / Figure 2.1 / Fig 2.1
        lab = re.escape(label)
        lab_alt = lab.replace(r'\.', r'\\.?')
        patterns.append(re.compile(rf"(?:{lab}|{lab_alt})", re.I))
        # bare number form e.g. 2.1 from "Fig. 2.1"
        num_m = re.search(r"(\d+\.\d+)", label)
        if num_m:
            n = re.escape(num_m.group(1))
            patterns.append(re.compile(rf"(?:Fig\.?|Figure)\s*{n}", re.I))
        # caption keywords (significant words)
        if caption:
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", caption) if w.lower() not in {
                "that", "this", "with", "from", "into", "using", "based", "show", "shows", "figure", "image"
            }]
            if words:
                # prefer longer distinctive tokens
                words = sorted(set(words), key=len, reverse=True)[:4]
                for w in words:
                    patterns.append(re.compile(re.escape(w), re.I))

        insert_at = None
        for pat in patterns:
            m = pat.search(html)
            if not m:
                continue
            # Find end of the containing block element after the match
            tail = html[m.end():]
            # Prefer closing </p>, </li>, or next heading
            close_m = re.search(r"</(?:p|li|blockquote)>", tail, flags=re.I)
            if close_m:
                insert_at = m.end() + close_m.end()
                break
            # Otherwise insert right after the match sentence (next period + space or tag)
            sent = re.search(r"[\.\!\?](?:\s|<)", tail)
            if sent:
                insert_at = m.end() + sent.end() - (1 if sent.group(0).endswith("<") else 0)
                break
            insert_at = m.end()
            break

        if insert_at is None:
            remaining.append(block)
        else:
            html = html[:insert_at] + block + html[insert_at:]
            placed.append(label)

    if remaining:
        html = html + "".join(remaining)
    return html


def generate_with_groq(prompt, context_text=""):
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY missing - get free key at console.groq.com")
    final_prompt = f"{context_text}\n\n{prompt}" if context_text else prompt

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Tempaper, a professional academic document writer producing university-level term papers and textbook-quality chapters.\n"
                    "Every output must read and look like a carefully researched, professionally written university textbook chapter or high-quality undergraduate/postgraduate academic research document — not a chatbot reply or generic essay.\n\n"
                    "ABSOLUTE OUTPUT RULES:\n"
                    "1. Return ONLY the section body. Zero prefaces, zero meta text, zero AI voice "
                    "(ban: As an AI, Here is, Certainly, Hope this helps, In this response, I will now, Below is).\n"
                    "2. ZERO repetition of any kind: no repeated ideas, sentences, paragraphs, conclusions, or figure labels. Each paragraph must contribute a distinct idea, explanation, argument, example or piece of evidence.\n"
                    "3. Human formal academic English that is precise, mature and readable. Concrete claims. Varied sentence length and structure. Not robotic or template-like. Avoid empty phrases such as 'Furthermore', 'Moreover', 'In addition', 'It is important to note'.\n"
                    "4. NUMBERING: under any heading, numbered items must run 1, 2, 3… in order. "
                    "Never label every item as 1. Objectives and research questions must be sequential.\n"
                    "5. Headings: markdown ## / ### with exact TOC numbers (example: ## 2.3 Energy and Momentum). "
                    "Do not repeat the heading wording in the first sentence of the body. Ensure content under each heading actually discusses that subject in depth.\n"
                    "6. Bold key terms on first use only with **term**. Sparse *italics* only when needed.\n"
                    "7. Equations on their own line in plain readable form (K = 1/2 m v^2 or KE = ½mv²). "
                    "NEVER LaTeX: no \\frac \\text \\( \\) \\[ \\] \\omega \\tfrac \\cdot etc.\n"
                    "8. Figures: at most one or two natural references using the exact label once "
                    "(As shown in Fig. 2.1, …). Never 'Fig. 2.1 Fig. 2.1'.\n"
                    "9. Do not invent citations, data, authors, years, statistics, research findings, sample sizes, or DOIs. No HTML. No code fences. "
                    "If a source cannot be verified from the provided research notes, do not present it as confirmed; mark for student verification if necessary.\n"
                    "10. Write as a careful student or supervisor preparing a formal dissertation — professional, print-ready, Word-document quality.\n\n"
                    "ACADEMIC DEPTH AND TEXTBOOK STANDARD (mandatory):\n"
                    "- Explain concepts thoroughly rather than merely mentioning them. Introduce important concepts before discussing complex arguments.\n"
                    "- Define technical terms accurately and in their proper academic context.\n"
                    "- Explain the background, development and significance of the subject.\n"
                    "- Present relevant theories, models, principles and scholarly perspectives where appropriate.\n"
                    "- Connect paragraphs logically so that ideas develop progressively. Explain why a point matters, not merely state that it matters.\n"
                    "- Use examples, illustrations and relevant evidence where appropriate. Demonstrate critical thinking, interpretation and academic reasoning.\n"
                    "- Where relevant, organize discussions with: introduction to the concept, definition and explanation, historical or theoretical background, major components or principles, how it works, relevant theories and scholarly interpretations, practical applications and examples, benefits/limitations/challenges, critical discussion and implications, and a concise concluding synthesis. Adapt this intelligently to the discipline and section; do not force it everywhere.\n"
                    "- A reader should finish each major discussion with a clear understanding of the subject, including its meaning, importance, applications and limitations.\n"
                    "- Avoid shallow writing, inflated paragraphs with little information, restating introductions under every subsection, unnecessary summaries after every paragraph, and artificially extending sections merely to reach a word count.\n"
                    "- When discussing scholarly literature: explain different perspectives, compare and contrast arguments, identify agreements and disagreements, discuss strengths and limitations of theories, relate to the topic, identify research gaps where supported, and explain implications. Distinguish description from critical evaluation.\n"
                    "- Adapt writing to the actual discipline (e.g., education: theories and classroom applications; computer science: principles, architecture, algorithms and limitations; business: management theories and organizational practices; etc.). Do not introduce unrelated technical concepts.\n"
                    "- Never invent real-world research activities, findings, surveys, interviews, experiments, or data. Distinguish proposed methodology, existing published research, student-supplied data, illustrative hypothetical examples, and actual analyzed findings.\n"
                    "- Generate original academic explanations. Vary sentence structure naturally. Optimize for genuine academic quality, source attribution, originality and student understanding.\n"
                    "- Maintain a consistent scholarly tone appropriate to the academic level. Use credible, traceable references that match the citation style. In-text citations must correspond to reference list entries.\n"
                    "- Chapters must have clear title, meaningful introduction, logical major headings, properly nested subsections with consistent numbering, detailed discussion under each heading, relevant examples and evidence, logical transitions, and a coherent chapter conclusion where appropriate. Do not create headings with only one or two sentences unless genuinely appropriate.\n"
                ),
            },
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.55,
        max_tokens=8192,
        top_p=0.9,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError("Empty Groq response")
    # Always post-process: strip LaTeX, fix 1./1./1. lists, collapse duplicate Fig labels
    return normalize_server_text(text.strip())


def normalize_server_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Strip common LaTeX math delimiters / commands so the paper stays Word-like
    text = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", text)
    text = re.sub(r"\\(?:frac|tfrac|dfrac)\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:text|mathrm|mathbf|textit|textbf)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:omega|alpha|beta|gamma|theta|phi|pi|mu|sigma|Delta|delta)\b", lambda m: {
        "omega": "ω", "alpha": "α", "beta": "β", "gamma": "γ", "theta": "θ",
        "phi": "φ", "pi": "π", "mu": "μ", "sigma": "σ", "Delta": "Δ", "delta": "δ",
    }.get(m.group(0)[1:], m.group(0)), text)
    text = text.replace("\\cdot", "·").replace("\\times", "×").replace("\\pm", "±")
    text = re.sub(r"\^{([^{}]+)}", r"^\1", text)
    text = re.sub(r"_{([^{}]+)}", r"_\1", text)
    text = re.sub(r"\\+", "", text)  # leftover backslash commands
    # Headings that were bold-only lines → markdown h3
    text = re.sub(
        r"^\s*\*\*\s*(\d+(?:\.\d+)*\s*[^\*]+?)\s*\*\*\s*$",
        r"### \1",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$", r"### \1", text, flags=re.MULTILINE)
    # Collapse repeated figure labels: "Fig. 2.1 Fig. 2.1" → "Fig. 2.1"
    text = re.sub(
        r"\b((?:Fig\.?|Figure)\s*\d+(?:\.\d+)*)\s+\1\b",
        r"\1",
        text,
        flags=re.I,
    )
    # Fix lists where every item was wrongly numbered "1." → sequential 1,2,3…
    lines = text.splitlines()
    fixed_lines = []
    idx = 0
    while idx < len(lines):
        m = re.match(r"^(\s*)1[.)]\s+(.+)$", lines[idx])
        if m:
            indent, body = m.group(1), m.group(2)
            block = [(indent, body)]
            jdx = idx + 1
            while jdx < len(lines):
                m2 = re.match(r"^(\s*)1[.)]\s+(.+)$", lines[jdx])
                if m2 and m2.group(1) == indent:
                    block.append((m2.group(1), m2.group(2)))
                    jdx += 1
                else:
                    break
            if len(block) >= 2:
                for n, (ind, b) in enumerate(block, 1):
                    fixed_lines.append(f"{ind}{n}. {b}")
                idx = jdx
                continue
        fixed_lines.append(lines[idx])
        idx += 1
    text = "\n".join(fixed_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_duplicate_sections(text: str, similarity_threshold: float = 0.92) -> str:
    """
    Remove exact repeated blocks and near-duplicate consecutive blocks.
    Helps prevent repetitive conclusions, summaries, and references.
    """
    if not text:
        return text

    normalized = normalize_server_text(text)
    raw_blocks = re.split(r"\n\s*\n+", normalized.strip())
    cleaned_blocks = []
    seen_exact = set()
    prev = None

    for block in raw_blocks:
        b = block.strip()
        if not b:
            continue

        exact_key = re.sub(r"\s+", " ", b).strip().lower()
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        if prev is not None:
            prev_norm = re.sub(r"\s+", " ", prev).strip()
            b_norm = re.sub(r"\s+", " ", b).strip()
            ratio = difflib.SequenceMatcher(None, prev_norm, b_norm).ratio()
            if ratio >= similarity_threshold:
                continue

        cleaned_blocks.append(b)
        prev = b

    out = "\n\n".join(cleaned_blocks).strip()
    return out if out else normalized


def _dedupe_repeated_headings(text):
    """Remove repeated identical heading lines without touching repeated prose."""
    seen = set()
    output = []
    for line in str(text or "").splitlines():
        is_heading = bool(re.match(r"^\s*#{1,4}\s+", line) or re.match(r"^\s*\*\*.+\*\*\s*$", line) or re.match(r"^\s*\d+(?:\.\d+)+[.)]?\s+", line))
        if is_heading:
            key = re.sub(r"^\s*#{1,4}\s*", "", line).strip().strip("*").strip()
            key = re.sub(r"\s+", " ", key).casefold()
            if key and key in seen:
                continue
            if key: seen.add(key)
        output.append(line)
    return "\n".join(output).strip()


def _chapter_quality_issues(text: str, title: str = "", toc_text: str = ""):
    """Cheap local checks; avoids extra model calls unless a chapter looks incomplete."""
    issues = []
    if not text or len(text.split()) < 120:
        issues.append("The chapter is unusually short and may be incomplete.")
    lines = text.splitlines()
    heading_re = re.compile(r"^\s*(?:#{1,3}\s*)?(?:\*\*)?(\d+\.\d+(?:\.\d+)?)[.)]?\s+(.+?)(?:\*\*)?\s*$")
    found = []
    for i, line in enumerate(lines):
        m = heading_re.match(line.strip())
        if m:
            found.append((i, m.group(1), m.group(2).strip()))
    for pos, (line_index, number, label) in enumerate(found):
        end = found[pos + 1][0] if pos + 1 < len(found) else len(lines)
        body = "\n".join(lines[line_index + 1:end]).strip()
        body_words = len(re.findall(r"\b[\w'-]+\b", body))
        # A heading with only a label, or a very short body, is a strong omission signal.
        if body_words < 28:
            issues.append(f"Subheading {number} ({label}) has only {body_words} words beneath it before the next numbered subheading.")
    # Compare the generated chapter against the subsection entries already specified in the saved TOC.
    chapter_number = None
    cm = re.search(r"chapter\s+(\d+|i{1,3}|iv|v|vi{0,3}|ix|x|one|two|three|four|five|six|seven|eight|nine|ten)\b", title, re.I)
    if cm:
        token = cm.group(1).lower()
        word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        roman_nums = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
        chapter_number = int(token) if token.isdigit() else word_nums.get(token, roman_nums.get(token))
    if chapter_number and toc_text:
        expected = []
        for toc_line in toc_text.splitlines():
            tm = re.match(r"^\s*(?:[-*]\s*)?(\d+)\.(\d+)(?:\s+|[.)])(.+?)\s*$", re.sub(r"[*#`]", "", toc_line))
            if tm and int(tm.group(1)) == chapter_number:
                expected.append((f"{tm.group(1)}.{tm.group(2)}", tm.group(3).strip()))
        actual_numbers = {number for _, number, _ in found}
        missing = [f"{number} {label}" for number, label in expected if number not in actual_numbers]
        if missing:
            issues.append("Required TOC subheading(s) missing from this chapter: " + "; ".join(missing) + ".")
    # Flag label-only list items only when the surrounding text explicitly announces
    # a list of items that should be discussed (not objectives or simple enumerations).
    for i, line in enumerate(lines):
        if re.search(r"\b(the following|include the following|following are|major (?:types|factors|causes|reasons|characteristics|elements|functions|advantages|disadvantages)|types of|factors affecting|causes of|reasons for|characteristics of)\b", line, re.I):
            window = "\n".join(lines[i:min(len(lines), i + 14)])
            items = re.findall(r"^\s*(?:\d+[.)]|[-*+])\s+(.+)$", window, re.M)
            if len(items) >= 2 and sum(len(re.findall(r"\b[\w'-]+\b", item)) < 10 for item in items) >= 2:
                issues.append("A list introduced as requiring discussion contains multiple label-only/very short items; explain each item where the context requires it.")
                break
    return issues


def _repair_incomplete_chapter(content: str, title: str, prompt_text: str, toc_text: str, research: str = ""):
    """Run a targeted repair only if deterministic checks find likely omissions."""
    issues = _chapter_quality_issues(content, title, toc_text)
    if not issues:
        return content, []
    repair_prompt = (
        f"Review and repair ONLY the existing chapter titled: {title}. Return the complete corrected chapter, not a report. "
        "Preserve correct material and the original heading names/order. Do not rewrite the whole paper or add unrelated sections. "
        "Elevate the academic quality to university textbook standard: explain concepts thoroughly, provide critical analysis, logical development, examples and evidence. "
        "Avoid repetition, shallow writing and padding. "
        "Check every numbered subheading has the content its topic requires. Explanatory subsections need a meaningful developed explanation; "
        "a calculation/solution section may show the method and answer without artificial padding. Where a list is introduced as types, factors, causes, "
        "reasons, characteristics, elements, functions, advantages/disadvantages, stages or methods requiring discussion, explain each item under its parent heading. "
        "Do not explain simple objective lists or short enumerations unnecessarily. Keep lists, examples, equations and their explanations under the correct subheading. "
        "Do not invent sources, citations, research findings or data.\n\nDetected issues:\n- " + "\n- ".join(issues) +
        f"\n\nSaved requirements:\n{prompt_text}\n\nSaved TOC:\n{toc_text}\n\nResearch notes (use only when relevant):\n{research}\n\nExisting chapter to repair:\n{content}"
    )
    repaired = _dedupe_repeated_headings(remove_duplicate_sections(generate_with_groq(repair_prompt)))
    # Do not replace the original with a much shorter failed repair.
    if len(repaired.split()) < max(100, int(len(content.split()) * 0.65)):
        return content, issues + ["Repair response was too short; original chapter retained."]
    return repaired, issues


# --- ROUTES (original term-paper logic preserved) ---

@app.route("/workflow-projects", methods=["GET"])
def workflow_projects():
    """List staged term-paper projects for Recent Papers. Display title is the topic."""
    uid = request.args.get("user_id")
    if not uid or not str(uid).isdigit():
        return jsonify({"projects": []}), 200
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.prompt_code, p.toc_code, p.current_index, p.merged, p.updated_at,
                       pr.title, t.sections_json
                FROM paper_projects p
                JOIN paper_prompts pr ON pr.prompt_code = p.prompt_code
                JOIN paper_tocs t ON t.toc_code = p.toc_code
                WHERE p.user_id = %s
                ORDER BY p.updated_at DESC NULLS LAST, p.id DESC
                LIMIT 40
                """,
                (int(uid),),
            )
            rows = cur.fetchall()
        projects = []
        for r in rows:
            sections = r[7] or []
            try:
                n_sections = len(sections) if isinstance(sections, list) else 0
            except Exception:
                n_sections = 0
            projects.append({
                "project_id": r[0],
                "prompt_code": r[1],
                "toc_code": r[2],
                "current_index": r[3],
                "merged": bool(r[4]),
                "updated_at": r[5].isoformat() if r[5] else None,
                "title": r[6] or "Untitled paper",
                "section_count": n_sections,
            })
        return jsonify({"projects": projects}), 200
    except Exception as e:
        logger.exception("workflow-projects failed")
        return jsonify({"projects": [], "error": str(e)[:200]}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ok", "message": f"Monkey Scout API running - Groq {GROQ_MODEL}"}), 200


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")
    if not username or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    conn = None
    try:
        conn = get_db()
        otp = generate_otp("ver")
        exp = datetime.utcnow() + timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username,email,password_hash,is_verified,otp_code,otp_expires_at) VALUES (%s,%s,%s,FALSE,%s,%s) RETURNING id",
                (username, email, generate_password_hash(password), otp, exp),
            )
            uid = cur.fetchone()[0]
            conn.commit()
        if not send_email(email, "Verify Your Monkey Scout Account", f"<p>Your code: <strong>{otp}</strong></p>"):
            return jsonify({"error": "Email could not be sent"}), 502
        return jsonify({"message": "Registered", "user_id": uid}), 201
    except psycopg2.IntegrityError:
        if conn:
            conn.rollback()
        return jsonify({"error": "Username or Email exists"}), 400
    finally:
        if conn:
            release_db(conn)


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    ident = str(data.get("identifier", "")).strip().lower()
    pwd = data.get("password", "")
    if not ident or not pwd:
        return jsonify({"error": "Missing fields"}), 400

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,username,email,password_hash,is_verified FROM users WHERE LOWER(email)=%s OR LOWER(username)=%s",
                (ident, ident),
            )
            r = cur.fetchone()

        if not r or not check_password_hash(r[3], pwd):
            return jsonify({"error": "Invalid credentials"}), 401
        if not r[4]:
            return jsonify({"error": "Verify email first"}), 403

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET last_login=%s WHERE id=%s",
                (datetime.utcnow(), r[0]),
            )
            conn.commit()

        return jsonify(
            {
                "message": "Login ok",
                "user_id": r[0],
                "user": {"id": r[0], "username": r[1], "email": r[2]},
            }
        ), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/verify-email", methods=["POST"])
def verify_email():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,otp_code,otp_expires_at FROM users WHERE LOWER(email)=%s", (email,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "User not found"}), 404
            if row[1] != code:
                return jsonify({"error": "Invalid code"}), 400
            if row[2] and datetime.utcnow() > row[2]:
                return jsonify({"error": "Code expired"}), 400
            cur.execute("UPDATE users SET is_verified=TRUE,otp_code=NULL,otp_expires_at=NULL WHERE id=%s", (row[0],))
            conn.commit()
        return jsonify({"message": "Verified"}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/resend-verification", methods=["POST"])
def resend():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    conn = None
    try:
        conn = get_db()
        otp = generate_otp("ver")
        exp = datetime.utcnow() + timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET otp_code=%s,otp_expires_at=%s WHERE LOWER(email)=%s RETURNING is_verified",
                (otp, exp, email),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Email not found"}), 404
            conn.commit()
        send_email(email, "Verification Code", f"<p>Code: <strong>{otp}</strong></p>")
        return jsonify({"message": "Code resent"}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/request-reset", methods=["POST"])
def req_reset():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    conn = None
    try:
        conn = get_db()
        otp = generate_otp("pass")
        exp = datetime.utcnow() + timedelta(minutes=15)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET otp_code=%s,otp_expires_at=%s WHERE LOWER(email)=%s RETURNING id",
                (otp, exp, email),
            )
            if not cur.fetchone():
                return jsonify({"message": "If email exists, code sent"}), 200
            conn.commit()
        send_email(email, "Reset Code", f"<h1>{otp}</h1>")
        return jsonify({"message": "Code sent"}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/reset-password", methods=["POST"])
def reset_pwd():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    np = data.get("new_password", "")
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT otp_code,otp_expires_at FROM users WHERE LOWER(email)=%s", (email,))
            r = cur.fetchone()
            if not r or r[0] != code:
                return jsonify({"error": "Invalid code"}), 400
            if r[1] and datetime.utcnow() > r[1]:
                return jsonify({"error": "Expired"}), 400
            cur.execute(
                "UPDATE users SET password_hash=%s,otp_code=NULL WHERE LOWER(email)=%s",
                (generate_password_hash(np), email),
            )
            conn.commit()
        return jsonify({"message": "Password updated"}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/me", methods=["GET"])
@cache.cached(timeout=300, query_string=True)
def get_me():
    uid = request.args.get("user_id")
    if not uid or not str(uid).isdigit():
        return jsonify({"user": {"id": 0, "username": "Guest", "email": "guest@monkeyscout.com"}}), 200

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT id,username,email,is_verified FROM users WHERE id=%s", (int(uid),))
            r = cur.fetchone()
            if not r:
                return jsonify({"error": "Not found"}), 404
        return jsonify({"user": {"id": r[0], "username": r[1], "email": r[2], "is_verified": r[3]}}), 200
    finally:
        if conn:
            release_db(conn)


# --- ORIGINAL generate route (TERM PAPER generation) - UNTOUCHED LOGIC ---
@app.route("/generate", methods=["POST"])
def generate_paper():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    user_id = data.get("user_id", "guest")
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    if len(prompt) > 20000:
        return jsonify({"error": "Prompt too long"}), 413
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY missing"}), 503

    try:
        tavily_context, sources = search_tavily(prompt)
        final_content = f"Topic: {prompt}\n\nReference Material:\n{tavily_context}" if tavily_context else prompt

        result_text = generate_with_groq(prompt, final_content)
        result_text = remove_duplicate_sections(result_text)

        word_count = len(result_text.split())
        page_count = max(1, word_count // 250)

        if str(user_id).isdigit():
            conn = None
            try:
                conn = get_db()
                sid = "".join(random.choices(string.ascii_letters + string.digits, k=32))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO paper_sessions (user_id,session_id,title,prompt_text,result_text,word_count,page_count) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (int(user_id), sid, prompt[:200], prompt, result_text, word_count, page_count),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Save fail: {e}")
            finally:
                if conn:
                    release_db(conn)

        return jsonify({"result": result_text, "sources": sources, "word_count": word_count, "page_count": page_count}), 200
    except Exception as e:
        logger.exception(f"Generate fail: {e}")
        msg = str(e)
        if "429" in msg:
            return jsonify({"error": "Groq rate limit, wait 10s"}), 429
        return jsonify({"error": "Generation failed", "details": msg[:300]}), 500


# --- Keep papers listing as before ---
@app.route("/papers", methods=["GET"])
def get_papers():
    uid = request.args.get("user_id")
    if not uid or not str(uid).isdigit():
        return jsonify({"papers": []}), 200

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,session_id,title,prompt_text,word_count,page_count,created_at FROM paper_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 50",
                (int(uid),),
            )
            rows = cur.fetchall()
        papers = [
            {
                "id": r[0],
                "session_id": r[1],
                "title": r[2],
                "prompt": r[3],
                "word_count": r[4],
                "page_count": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
        return jsonify({"papers": papers}), 200
    finally:
        if conn:
            release_db(conn)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    uid = request.args.get("user_id") or (request.get_json(silent=True) or {}).get("user_id")
    if not uid or not str(uid).isdigit():
        return jsonify({"error": "Valid user_id required"}), 400

    if request.method == "GET":
        conn = None
        try:
            conn = get_db()
            with conn.cursor() as cur:
                cur.execute("SELECT theme,font_size,notifications,auto_save FROM user_settings WHERE user_id=%s", (int(uid),))
                r = cur.fetchone()
            if not r:
                return jsonify({"settings": {"theme": "system", "font_size": 16, "notifications": True, "auto_save": True}}), 200
            return jsonify({"settings": {"theme": r[0], "font_size": r[1], "notifications": r[2], "auto_save": r[3]}}), 200
        finally:
            if conn:
                release_db(conn)

    data = request.get_json(silent=True) or {}
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_settings (user_id,theme,font_size,notifications,auto_save) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO UPDATE SET theme=EXCLUDED.theme,font_size=EXCLUDED.font_size,notifications=EXCLUDED.notifications,auto_save=EXCLUDED.auto_save",
                (
                    int(uid),
                    data.get("theme", "system"),
                    data.get("font_size", 16),
                    data.get("notifications", True),
                    data.get("auto_save", True),
                ),
            )
            conn.commit()
        return jsonify({"message": "Updated"}), 200
    finally:
        if conn:
            release_db(conn)


# --- General AI assistant endpoint (does NOT replace /generate) ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    user_id = data.get("user_id", "guest")
    save = bool(data.get("save", False))
    system_prompt = data.get("system_prompt", None)
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 1024))

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY missing"}), 503

    try:
        tavily_context, sources = search_tavily(prompt)
        final_content = f"{tavily_context}\n\n{prompt}" if tavily_context else prompt

        system_msg = system_prompt if system_prompt else "You are a helpful, concise assistant. Answer user questions clearly and honestly. If you don't know, say you don't know."

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": final_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        text = None
        try:
            text = response.choices[0].message.content
        except Exception:
            try:
                text = response.choices[0].text
            except Exception:
                text = str(response)

        if not text or not str(text).strip():
            raise RuntimeError("Empty Groq response")

        raw_result = str(text).strip()
        cleaned_result = remove_duplicate_sections(raw_result)
        word_count = len(cleaned_result.split())

        if save and str(user_id).isdigit():
            conn = None
            try:
                conn = get_db()
                sid = "".join(random.choices(string.ascii_letters + string.digits, k=32))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO chat_sessions (user_id,session_id,title,prompt_text,result_text,word_count) VALUES (%s,%s,%s,%s,%s,%s)",
                        (int(user_id), sid, prompt[:200], prompt, cleaned_result, word_count),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Chat save fail: {e}")
            finally:
                if conn:
                    release_db(conn)

        return jsonify({"result": cleaned_result, "sources": sources, "word_count": word_count}), 200
    except Exception as e:
        logger.exception(f"Chat fail: {e}")
        msg = str(e)
        if "429" in msg:
            return jsonify({"error": "Groq rate limit, wait"}), 429
        return jsonify({"error": "Chat generation failed", "details": msg[:300]}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
