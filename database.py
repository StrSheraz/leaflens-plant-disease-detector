"""
database.py — Disease History SQLite Manager
Apne project folder mein rakho (app.py ke saath)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "history.db"


def init_db():
    """Database aur table banao agar exist nahi karta."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            disease     TEXT    NOT NULL,
            filename    TEXT    NOT NULL,
            pred_index  INTEGER NOT NULL,
            confidence  INTEGER,
            detected_at TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_detection(disease: str, filename: str, pred_index: int, confidence: int = None):
    """Ek nai detection save karo."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO detections (disease, filename, pred_index, confidence, detected_at) VALUES (?,?,?,?,?)",
        (disease, filename, pred_index, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_all_detections(limit: int = 100):
    """Sari detections lo, newest pehle."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_disease_counts():
    """Har disease ki count lo — chart ke liye."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT disease, COUNT(*) as count
        FROM detections
        GROUP BY disease
        ORDER BY count DESC
        LIMIT 10
    """)
    rows = c.fetchall()
    conn.close()
    return [{"disease": r[0], "count": r[1]} for r in rows]


def get_daily_counts(days: int = 14):
    """Last N days ki daily detection counts."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT DATE(detected_at) as day, COUNT(*) as count
        FROM detections
        WHERE detected_at >= DATE('now', ?)
        GROUP BY day
        ORDER BY day ASC
    """, (f"-{days} days",))
    rows = c.fetchall()
    conn.close()
    return [{"day": r[0], "count": r[1]} for r in rows]


def get_stats():
    """Summary stats."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT disease) FROM detections")
    unique = c.fetchone()[0]
    c.execute("SELECT disease FROM detections GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 1")
    top = c.fetchone()
    conn.close()
    return {
        "total": total,
        "unique_diseases": unique,
        "top_disease": top[0] if top else "—"
    }


def clear_history():
    """Saari history delete karo."""
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM detections")
    conn.commit()
    conn.close()