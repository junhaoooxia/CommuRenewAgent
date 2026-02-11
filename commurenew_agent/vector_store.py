from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

import numpy as np

from .models import KnowledgeNode, RetrievedNode


class SQLiteVectorStore:
    def __init__(self, db_path: str | Path = "data/knowledge.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                main_text TEXT NOT NULL,
                images_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert_node(self, node: KnowledgeNode, embedding: np.ndarray) -> None:
        self.conn.execute(
            """
            INSERT INTO knowledge_nodes (id, type, title, main_text, images_json, metadata_json, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type,
                title=excluded.title,
                main_text=excluded.main_text,
                images_json=excluded.images_json,
                metadata_json=excluded.metadata_json,
                embedding=excluded.embedding
            """,
            (
                node.id,
                node.type,
                node.title,
                node.main_text,
                json.dumps(node.images, ensure_ascii=False),
                json.dumps(node.metadata, ensure_ascii=False),
                embedding.astype(np.float32).tobytes(),
            ),
        )
        self.conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int = 15) -> List[RetrievedNode]:
        rows = self.conn.execute(
            "SELECT id, type, title, main_text, images_json, metadata_json, embedding FROM knowledge_nodes"
        ).fetchall()
        matches: List[RetrievedNode] = []
        for row in rows:
            emb = np.frombuffer(row[6], dtype=np.float32)
            score = float(np.dot(query_embedding, emb))
            matches.append(
                RetrievedNode(
                    id=row[0],
                    type=row[1],
                    title=row[2],
                    text=row[3],
                    images=json.loads(row[4]),
                    score=score,
                    metadata=json.loads(row[5]),
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    def close(self) -> None:
        self.conn.close()
