"""
database.py — saves every prediction and the user's feedback in a small SQLite database.

SQLite = a whole database inside ONE file (database/predictions.db). Python has it built in,
so there is nothing to install or start.

Two tables:
    predictions : one row per analysed photo
    feedback    : the user says if the answer was right (linked to a prediction by prediction_id)

Used by: app/streamlit_app.py and app/api.py.
"""
import json
import sqlite3
from datetime import datetime

import config


def connect(database_file=None):
    """Open the database file (default: config.DATABASE_FILE). The folder is created if needed."""
    if database_file is None:
        database_file = config.DATABASE_FILE
    database_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_file)
    connection.row_factory = sqlite3.Row          # rows behave like dictionaries: row["decision"]
    return connection


def create_tables(database_file=None):
    with connect(database_file) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT,
                image_name      TEXT,
                top_class       TEXT,
                top_probability REAL,
                decision        TEXT,     -- advise / refer / unknown
                advised_group   TEXT,
                prediction_set  TEXT,     -- list of groups, stored as JSON text
                energy          REAL
            )""")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES predictions(id),
                created_at    TEXT,
                is_correct    INTEGER,    -- 1 = yes, 0 = no
                true_class    TEXT,
                comment       TEXT
            )""")


def save_prediction(image_name, result, database_file=None):
    """Store one result from PlantDiseasePredictor.predict(). Returns the new row's id."""
    top = result["top_classes"][0]
    with connect(database_file) as connection:
        cursor = connection.execute(
            "INSERT INTO predictions (created_at, image_name, top_class, top_probability, decision, "
            "advised_group, prediction_set, energy) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), image_name, top["class_name"], top["probability"],
             result["decision"], result["advised_group"], json.dumps(result["prediction_set"]), result["energy"]))
        return cursor.lastrowid


def save_feedback(prediction_id, is_correct, true_class="", comment="", database_file=None):
    with connect(database_file) as connection:
        connection.execute(
            "INSERT INTO feedback (prediction_id, created_at, is_correct, true_class, comment) VALUES (?, ?, ?, ?, ?)",
            (prediction_id, datetime.now().isoformat(timespec="seconds"), int(is_correct), true_class, comment))


def get_history(limit=50, database_file=None):
    """The latest predictions, newest first, with their feedback (if any)."""
    with connect(database_file) as connection:
        rows = connection.execute(
            "SELECT p.id, p.created_at, p.image_name, p.top_class, p.top_probability, p.decision, "
            "p.advised_group, p.prediction_set, f.is_correct "
            "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id "
            "ORDER BY p.id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
