import json
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("DATABASE_URL", BASE_DIR / "database" / "app.db"))


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            details TEXT,
            face_width REAL DEFAULT 0,
            face_height REAL DEFAULT 0,
            forehead_width REAL DEFAULT 0,
            jaw_width REAL DEFAULT 0
        )
        """
    )

    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "image_path" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN image_path TEXT")
    if "details" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN details TEXT")
    if "face_width" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN face_width REAL DEFAULT 0")
    if "face_height" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN face_height REAL DEFAULT 0")
    if "forehead_width" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN forehead_width REAL DEFAULT 0")
    if "jaw_width" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN jaw_width REAL DEFAULT 0")

    conn.commit()
    conn.close()


def save_prediction(filename, prediction, confidence, image_path=None, metrics=None):
    conn = get_connection()
    details = json.dumps(metrics or {}, ensure_ascii=False)
    cursor = conn.execute(
        """
        INSERT INTO predictions (filename, prediction, confidence, image_path, details, face_width, face_height, forehead_width, jaw_width)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            prediction,
            confidence,
            image_path,
            details,
            (metrics or {}).get("face_width", 0),
            (metrics or {}).get("face_height", 0),
            (metrics or {}).get("forehead_width", 0),
            (metrics or {}).get("jaw_width", 0),
        ),
    )
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id


def get_predictions(limit=10):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, filename, prediction, confidence, created_at, image_path, details, face_width, face_height, forehead_width, jaw_width
        FROM predictions ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id, filename, prediction, confidence, created_at, image_path, details, face_width, face_height, forehead_width, jaw_width
        FROM predictions WHERE id = ?
        """,
        (prediction_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_prediction(prediction_id):
    conn = get_connection()
    conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()
