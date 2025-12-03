from __future__ import annotations
from typing import List, Tuple
import sqlite3
import json
import numpy as np


class Database:
    """SQLite storage for persons and face embeddings."""

    def __init__(self, path: str = "surveillance.db") -> None:
        self.path = path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id)
                )
                """
            )
            con.commit()

    def add_person_with_embedding(self, name: str, embedding: np.ndarray) -> int:
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            cur.execute("INSERT OR IGNORE INTO persons(name) VALUES (?)", (name,))
            cur.execute("SELECT id FROM persons WHERE name=?", (name,))
            pid = int(cur.fetchone()[0])
            emb_json = json.dumps(embedding.astype(float).tolist())
            cur.execute(
                "INSERT INTO embeddings(person_id, embedding_json) VALUES (?,?)",
                (pid, emb_json),
            )
            con.commit()
            return pid

    def get_all_embeddings(self) -> Tuple[List[str], List[np.ndarray]]:
        labels = []
        embs = []
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT p.name, e.embedding_json
                FROM persons p
                JOIN embeddings e ON p.id = e.person_id
                """
            )
            for name, emb_str in cur.fetchall():
                arr = np.array(json.loads(emb_str), dtype=float)
                labels.append(name)
                embs.append(arr)
        return labels, embs

    def log_zone_violation(self, track_id: int, zone_name: str, 
                          person_id: int = None) -> None:
        """Log a zone violation event."""
        with sqlite3.connect(self.path) as con:
            cur = con.cursor()
            
            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zone_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    zone_name TEXT NOT NULL,
                    person_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                INSERT INTO zone_violations (track_id, zone_name, person_id)
                VALUES (?, ?, ?)
            """, (track_id, zone_name, person_id))
            
            con.commit()
