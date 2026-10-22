import os
import json
import re
import random
import string
import logging
import functools
import difflib
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
    origins=[
        "https://odogid101-cmd.github.io",
        "http://localhost:3000",
        "http://localhost:5000",
    ],
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
        conn.commit()
    except Exception:
        if conn: conn.rollback()
        logger.exception("Workflow DB initialization failed")
    finally:
        if conn: release_db(conn)

init_workflow_db()


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


def _section_items(toc_text, config):
    headings = []
    for line in (toc_text or "").splitlines():
        line = re.sub(r"[*#`]", "", line).strip()
        line = re.sub(r"\s+\.{2,}\s*[ivxlcdm\d]+\s*$", "", line, flags=re.I)
        if not line or len(line) > 240: continue
        low = line.lower()
        if low in {"table of contents", "contents"}: continue
        if re.match(r"^(chapter\s+[ivxlcdm\d]+|chapter\s+\w+|(?:\d+|[ivxlcdm]+)[.)]?\s+\w)", line, re.I) or any(
            word in low for word in ("conclusion", "recommendation", "references", "bibliography", "appendix")):
            if not any(h.lower() == line.lower() for h in headings): headings.append(line)
    chapters = int(config.get("chapters") or 5)
    if not any(re.search(r"chapter\s+one|chapter\s+i\b", h, re.I) for h in headings):
        headings = [f"CHAPTER {n}" for n in range(1, chapters + 1)] + ["Conclusion", "References"]
    final = []
    for heading in headings:
        low = heading.lower()
        if any(x in low for x in ("certification", "dedication", "acknowledg", "abstract", "title page", "cover page")):
            continue
        if not any(h.lower() == low for h in final): final.append(heading)
    return [{"key": re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-")[:90], "title": h} for h in final]


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
GENERAL PAGE RULE: No chapter may be shorter than 3 A4 pages. Match actual depth to topic weight: LIGHT 3–4 pages, MEDIUM 5–7 pages, HEAVY 8–12 pages. Treat these as content-depth targets, not fabricated page-count claims. Do not make all chapters follow the same pattern.
CHAPTER 1 (opening/introduction/background): LIGHT, 3–4 pages; 2–3 numbered subheadings; exactly one simple numbered list for objectives/questions; no nested A/B list.
MIDDLE CHAPTERS: rotate structure naturally. A heavy Chapter 2 (literature/conceptual review) should have 4–5 subheadings, two simple lists, one compulsory nested A/B list, and substantial depth targeting 8–12 pages. A medium Chapter 3 should have 3 subheadings, two simple lists, one nested A/B list, targeting 5–7 pages. A light/medium Chapter 4 should have 2–3 subheadings, one or two simple lists, and no nested list unless genuinely useful (maximum one). For further middle chapters rotate 4 headings, then 2, then 3; adapt to the topic rather than forcing irrelevant content.
FINAL CHAPTER: 3–4 pages; exactly three subheadings titled Summary, Conclusion, and Recommendation/Recommendations; one plain numbered recommendations list; no nested list.
SUBHEADING STYLE AND COMPLETENESS: use bold numbered subheadings (e.g. 2.1, 2.2) exactly as listed in the saved TOC and in the same order. Each explanatory subheading must contain a developed, topic-specific explanation (normally at least 5–7 substantial displayed lines or equivalent detail, but do not pad with repetition). Classify each subsection by what the topic requires: explanation, list, example, comparison, procedure, calculation/solution, or a combination. Do not force long prose into a subsection whose correct answer is a concise calculation or direct result.
LIST EXPLANATION RULE: when the prose introduces a list as types, factors, causes, reasons, characteristics, elements, functions, advantages/disadvantages, stages, methods, or similar items that require discussion, explain EACH list item after naming it. A label-only list is incomplete when the context requires explanation. Each item should normally include a clear sentence explaining what it means, why it matters, or how it applies, plus an example when useful. Keep list items under the correct parent subheading; never let list items become accidental numbered headings. Not every list needs expanded explanations: contents pages, objectives, short factual answers, and simple enumerations may remain concise when academically appropriate. Finish the current subheading completely before starting the next one.
Simple lists use plain 1, 2, 3 numbering, not bold. Nested lists use bold A. and B. labels with explanations only where a parent point has distinct types; at most one nested A/B list in any chapter. Do not add lists merely to satisfy a count if they make the discussion unnatural; preserve the closest applicable pattern and academic relevance.
COVER PAGE: include the topic in a prominent centered bold heading and all available metadata below. For every missing item, print a clear blank line such as ____________________, never invent personal/institution/course details.
PRELIMINARY PAGES: arrange each page heading centered and bold; body paragraphs justified. Use conventional academic order where appropriate: cover/title page, certification/approval, declaration, dedication, acknowledgements, abstract, table of contents, list of tables/figures/abbreviations if relevant. Cover page has no visible page number; preliminary pages use lower-case Roman numerals; main chapters use Arabic numerals. Keep preliminary headings centered and bold, and body paragraphs justified.
TABLE OF CONTENTS: include preliminary items, every chapter and its subheadings, conclusion/recommendations, references and appendices when relevant. Do not invent page numbers in generated prose. The app will recalculate TOC page numbers from the current rendered/paginated document after each Pull Next or edit/regeneration; preserve TOC entry wording and update matching page values rather than creating a new TOC code.
ACADEMIC QUALITY: formal clear language, meaningful transitions, topic-specific discussion, credible sources and consistent citations. Do not fabricate sources, quotations, data, or claims. Each next pull must follow this full rule set plus the saved TOC and current chapter's role.'''
    prompt_text=(f"Term-paper topic: {title}\nTarget length: {config['pages']} A4 pages.\nRequired chapters: {config['chapters']}.\n"
        f"Cover-page details (leave a blank line for any missing value):\n{cover_lines}\n"
        f"Body font: {config['body_font_size']} pt; Heading 1: {config['heading1_size']} pt ({'bold' if config.get('heading1_bold',True) else 'regular'}); "
        f"Heading 2: {config['heading2_size']} pt ({'bold' if config.get('heading2_bold',True) else 'regular'}); "
        f"Heading 3: {config['heading3_size']} pt ({'bold' if config.get('heading3_bold',True) else 'regular'}).\n"
        f"Structure and preliminary pages: {config.get('structure','Standard academic structure')}.\n"
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
        prompt=("Create a complete, logically ordered academic term-paper TABLE OF CONTENTS only. Include required preliminary pages, exactly the requested number of chapters, useful numbered chapter subsections, conclusion/recommendations/references/appendices where appropriate. Reflect the automatic varied chapter-pattern rules in the saved prompt. Do not write the paper. Use clean Markdown, one entry per line, and do not invent page numbers.\n\n"
                f"Saved instructions:\n{prompt_text}\n\nTopic: {title}\nRequested chapters: {config.get('chapters',5)}")
        toc_text=normalize_server_text(generate_with_groq(prompt))
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
            cur.execute("SELECT id,current_index FROM paper_projects WHERE user_id=%s AND prompt_code=%s AND toc_code=%s",(int(uid),pc,tc)); project=cur.fetchone()
        if project:
            with conn.cursor() as cur:
                cur.execute("SELECT order_no,section_title,content_text FROM paper_sections WHERE project_id=%s ORDER BY order_no",(project[0],)); rows=cur.fetchall()
            return jsonify({"message":"Resumed your saved paper.","project_id":project[0],"current_index":project[1],
                            "sections":[{"order_no":r[0],"title":r[1],"content":r[2]} for r in rows]}),200
        _,title,config,prompt_text=pr; _,_,toc_text,sections,_=toc
        research,sources=search_tavily(prompt_text)
        prompt=("Write ONLY the cover/title page and required preliminary pages, ending with the complete Table of Contents. Do NOT write Chapter One or any chapter content. Follow every automatic formatting rule in the saved prompt. COVER PAGE: center the full topic in bold; include institution, faculty, department, course title/code, student name, registration number, lecturer/supervisor, academic session, and submission date. For each missing value, leave a visible blank line (____________________); do not invent details. Each preliminary-page heading must be centered and bold; body paragraphs must be justified. Put the cover first with no visible page number; use lower-case Roman numeral pagination for following preliminary pages. The app will update TOC page numbers as pages are rendered, so do not invent page numbers in generated text.\n\n"
                f"SAVED PROMPT:\n{prompt_text}\n\nSAVED TABLE OF CONTENTS:\n{toc_text}\n\nResearch notes:\n{research}\n\nWrite the preliminary material now.")
        prelim=remove_duplicate_sections(generate_with_groq(prompt))
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_projects (user_id,prompt_code,toc_code,current_index) VALUES (%s,%s,%s,0) RETURNING id",(int(uid),pc,tc)); pid=cur.fetchone()[0]
            cur.execute("INSERT INTO paper_sections (project_id,order_no,section_key,section_title,content_text,word_count) VALUES (%s,0,'preliminary','Preliminary Pages and Table of Contents',%s,%s)",(pid,prelim,len(prelim.split())))
        conn.commit()
        return jsonify({"message":"Preliminary pages created. Chapter One has not been generated yet.","project_id":pid,"current_index":0,
                        "result":prelim,"toc":toc_text,"toc_code":tc,"sources":sources,"word_count":len(prelim.split()),
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
        if action=="edit":
            edited=str(data.get("content","")).strip(); order_no=int(data.get("order_no",current_index))
            if not edited:return jsonify({"error":"Edited content cannot be empty."}),400
            with conn.cursor() as cur:
                cur.execute("UPDATE paper_sections SET content_text=%s,word_count=%s,updated_at=CURRENT_TIMESTAMP WHERE project_id=%s AND order_no=%s RETURNING section_title",(edited,len(edited.split()),int(pid),order_no)); row=cur.fetchone()
                if not row:return jsonify({"error":"Section not found."}),404
                cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s",(tc,))
            conn.commit(); return jsonify({"message":"Section updated.","title":row[0],"result":edited,"order_no":order_no}),200
        if action=="resend":
            order_no=int(data.get("order_no",current_index))
            with conn.cursor() as cur:
                cur.execute("SELECT section_title FROM paper_sections WHERE project_id=%s AND order_no=%s",(int(pid),order_no)); row=cur.fetchone()
            if not row:return jsonify({"error":"There is no saved section to regenerate."}),404
            title=row[0]; prompt=(f"Regenerate ONLY this existing section titled {title}. Do not repeat other chapters or preliminary pages. Apply every automatic Tempaper rule in the saved instructions, including chapter role, topic-weight page-depth target, varied heading/list pattern, and numbering. Preserve the section's position in the saved TOC.\n\n{prompt_text}\n\nTOC:\n{toc_text}\n\nSection: {title}")
            content=remove_duplicate_sections(generate_with_groq(prompt))
            with conn.cursor() as cur:
                cur.execute("UPDATE paper_sections SET content_text=%s,word_count=%s,updated_at=CURRENT_TIMESTAMP WHERE project_id=%s AND order_no=%s",(content,len(content.split()),int(pid),order_no))
                cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s",(tc,))
            conn.commit(); return jsonify({"message":"Section regenerated without adding a duplicate.","title":title,"result":content,"order_no":order_no}),200
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
        chapter_titles = [str(x.get("title") or "") for x in sections if re.search(r"chapter\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", str(x.get("title") or ""), re.I)]
        is_last_chapter = bool(chapter_titles and title.strip().lower() == chapter_titles[-1].strip().lower())
        if any(term in title_low for term in ("references", "bibliography", "appendix", "appendices")):
            chapter_role = "supplementary section"
        elif re.search(r"chapter\s*(1|i|one)\b|introduction|background", title, re.I):
            chapter_role = "opening chapter"
        elif is_last_chapter or re.search(r"conclusion|recommendations?$", title, re.I):
            chapter_role = "closing chapter"
        else:
            chapter_role = "middle chapter"
        prompt=(f"Write ONLY the next section titled exactly: {title}. Do not write any other chapter or repeat content already generated. Apply ALL automatic Tempaper rules in the saved prompt. Current role: {chapter_role}. Determine topic weight from the actual subject and follow the corresponding page-depth target and varied chapter pattern. Use the saved TOC as the authoritative heading checklist: include every relevant subsection for this chapter in the exact order and preserve its wording. Finish each subsection before starting the next. Identify whether each subsection needs explanation, list, example, comparison, calculation/solution, or a combination. When a list is introduced as types, factors, causes, reasons, characteristics, elements, functions, advantages/disadvantages, stages, or methods that need discussion, explain EACH item under that same subsection; do not leave label-only items. Do not force extra prose into direct-answer/calculation sections or simple objective lists. Use numbered bold subheadings, appropriate plain numbered lists and at most one nested A/B list when required by the chapter pattern. Keep the chapter substantive; do not pad with repetition. Use academic tone and credible citations. Do not add a new TOC or fabricate page numbers.\n\n"
                f"Original saved instructions (mandatory on every pull):\n{prompt_text}\n\nSaved table of contents:\n{toc_text}\n\nSection {next_index} of {len(sections)}. Write this section only.")
        research,sources=search_tavily(f"{prompt_text}\nSection: {title}")
        content=remove_duplicate_sections(generate_with_groq(prompt,research))
        # Only pay for a second model call when local checks identify likely omissions.
        if re.search(r"chapter\s*(?:[1-9]|i\b|one\b)", title, re.I) or chapter_role in {"opening chapter", "middle chapter", "closing chapter"}:
            content, quality_issues = _repair_incomplete_chapter(content, title, prompt_text, toc_text, research)
            if quality_issues:
                logger.info("Chapter quality repair for %s: %s", title, "; ".join(quality_issues))
        with conn.cursor() as cur:
            cur.execute("INSERT INTO paper_sections (project_id,order_no,section_key,section_title,content_text,word_count) VALUES (%s,%s,%s,%s,%s,%s)",(int(pid),next_index,key,title,content,len(content.split())))
            cur.execute("UPDATE paper_projects SET current_index=%s,updated_at=CURRENT_TIMESTAMP WHERE id=%s",(next_index,int(pid)))
            cur.execute("UPDATE paper_tocs SET updated_at=CURRENT_TIMESTAMP WHERE toc_code=%s",(tc,))
        conn.commit()
        return jsonify({"message":f"{title} generated.","title":title,"result":content,"order_no":next_index,"current_index":next_index,
                        "sources":sources,"word_count":len(content.split()),"page_count":max(1,(len(content.split())+249)//250),"toc_code":tc,"toc":toc_text}),201
    except Exception as e:
        if conn: conn.rollback()
        logger.exception("Pull next failed")
        return jsonify({"error":"Could not generate the next section.","details":str(e)[:200]}),500
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
                match=None
                for title, info in page_map.items():
                    if plain.lower().startswith(str(title).lower()):
                        match=info
                        break
                if match is not None:
                    page=match.get("page_start","")
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
                    "You are Tempaper, an expert academic writing assistant. "
                    "Write a clear, well-structured academic paper based on the user's request. "
                    "Use a professional academic tone, strong thesis development, coherent structure, "
                    "and grounded reasoning. Avoid repeating headings, paragraphs, sections, or references. "
                    "Keep each section unique and concise, with smooth transitions."
                ),
            },
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.7,
        max_tokens=8192,
        top_p=0.9,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError("Empty Groq response")
    return text.strip()


def normalize_server_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(
        r"^\s*\*\*\s*(\d+(?:\.\d+)*\s*[^\*]+?)\s*\*\*\s*$",
        r"### \1",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$", r"### \1", text, flags=re.MULTILINE)
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
        "Check every numbered subheading has the content its topic requires. Explanatory subsections need a meaningful developed explanation; "
        "a calculation/solution section may show the method and answer without artificial padding. Where a list is introduced as types, factors, causes, "
        "reasons, characteristics, elements, functions, advantages/disadvantages, stages or methods requiring discussion, explain each item under its parent heading. "
        "Do not explain simple objective lists or short enumerations unnecessarily. Keep lists, examples, equations and their explanations under the correct subheading. "
        "Do not invent sources or citations.\n\nDetected issues:\n- " + "\n- ".join(issues) +
        f"\n\nSaved requirements:\n{prompt_text}\n\nSaved TOC:\n{toc_text}\n\nResearch notes (use only when relevant):\n{research}\n\nExisting chapter to repair:\n{content}"
    )
    repaired = remove_duplicate_sections(generate_with_groq(repair_prompt))
    # Do not replace the original with a much shorter failed repair.
    if len(repaired.split()) < max(100, int(len(content.split()) * 0.65)):
        return content, issues + ["Repair response was too short; original chapter retained."]
    return repaired, issues


# --- ROUTES (original term-paper logic preserved) ---
@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ok", "message": f"Tempaper API running - Groq {GROQ_MODEL}"}), 200


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
        if not send_email(email, "Verify Your Tempaper Account", f"<p>Your code: <strong>{otp}</strong></p>"):
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
        return jsonify({"user": {"id": 0, "username": "Guest", "email": "guest@tempaper.com"}}), 200

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
