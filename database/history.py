"""
Database Module — SQLite Diagnosis History
==========================================
Manages persistent storage of completed diagnostic sessions using SQLite.

Design decisions:
- Schema is created automatically if the database does not exist.
- Parameterized queries throughout to prevent SQL injection.
- Each session is stored as a single record with JSON-serialized fields.
- The stored inference snapshot is self-contained so history remains
  readable even if the knowledge base rules change later.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "history.db")


def get_db_path() -> str:
    """Return the database path, creating the directory if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return DB_PATH


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a database connection with row_factory for dict-like access."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> None:
    """
    Create the database schema if it does not already exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_sessions (
                session_id      TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                goal_id         TEXT NOT NULL,
                goal_label      TEXT NOT NULL,
                conclusion_id   TEXT,
                conclusion_label TEXT NOT NULL,
                status_label    TEXT NOT NULL,
                confidence      TEXT NOT NULL,
                questions_asked INTEGER NOT NULL DEFAULT 0,
                answers_json    TEXT NOT NULL,
                result_json     TEXT NOT NULL,
                inference_summary TEXT NOT NULL
            )
        """)
        conn.commit()


def save_session(session: dict, result: dict, db_path: str | None = None) -> str:
    """
    Save a completed diagnostic session to the database.
    
    Args:
        session: The inference engine session dict.
        result:  The result dict from finalize_session().
        db_path: Optional path override (used in testing).
        
    Returns:
        The session_id string of the saved record.
    """
    init_db(db_path)
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat(timespec="seconds")

    answers_log = session.get("answers_log", [])
    answers_json = json.dumps(answers_log)

    result_json = json.dumps(result)

    # Build a self-contained inference summary snapshot
    inference_summary = {
        "goal_id": result.get("goal_id"),
        "goal_label": result.get("goal_label"),
        "conclusion_id": result.get("conclusion_id"),
        "conclusion_label": result.get("conclusion_label"),
        "conclusion_description": result.get("conclusion_description"),
        "status_label": result.get("status_label"),
        "status_description": result.get("status_description"),
        "confidence": result.get("confidence"),
        "rules_used": result.get("rules_used", []),
        "fired_rules": result.get("fired_rules", []),
        "known_facts": result.get("known_facts", []),
        "next_steps": result.get("next_steps", []),
        "safety_note": result.get("safety_note", ""),
    }
    inference_summary_json = json.dumps(inference_summary)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO diagnosis_sessions
                (session_id, created_at, goal_id, goal_label,
                 conclusion_id, conclusion_label, status_label, confidence,
                 questions_asked, answers_json, result_json, inference_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                created_at,
                result.get("goal_id", ""),
                result.get("goal_label", ""),
                result.get("conclusion_id"),
                result.get("conclusion_label", "Unable to Determine"),
                result.get("status_label", ""),
                result.get("confidence", ""),
                len(answers_log),
                answers_json,
                result_json,
                inference_summary_json,
            ),
        )
        conn.commit()

    return session_id


def get_all_sessions(db_path: str | None = None) -> list[dict]:
    """
    Retrieve all saved sessions, most recent first.
    
    Returns a list of summary dicts (no large JSON blobs).
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT session_id, created_at, goal_label, conclusion_label,
                   status_label, confidence, questions_asked
            FROM diagnosis_sessions
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_detail(session_id: str, db_path: str | None = None) -> dict | None:
    """
    Retrieve the full detail of a single saved session.
    
    Returns a dict with all stored fields, or None if not found.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM diagnosis_sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    if not row:
        return None

    record = dict(row)
    # Deserialize JSON fields
    try:
        record["answers"] = json.loads(record["answers_json"])
    except Exception:
        record["answers"] = []
    try:
        record["result"] = json.loads(record["result_json"])
    except Exception:
        record["result"] = {}
    try:
        record["inference_summary"] = json.loads(record["inference_summary"])
    except Exception:
        record["inference_summary"] = {}

    return record


def delete_session(session_id: str, db_path: str | None = None) -> bool:
    """
    Delete a single session record.
    
    Returns True if a record was deleted, False if not found.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM diagnosis_sessions WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
    return cursor.rowcount > 0


def delete_all_sessions(db_path: str | None = None) -> int:
    """
    Delete all session records.
    
    Returns the number of records deleted.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM diagnosis_sessions")
        conn.commit()
    return cursor.rowcount


def count_sessions(db_path: str | None = None) -> int:
    """Return the total number of saved sessions."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM diagnosis_sessions"
        ).fetchone()
    return row["cnt"] if row else 0
