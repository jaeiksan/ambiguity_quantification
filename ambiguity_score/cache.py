import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class CacheRecord:
    key: str
    value: Any
    created_at: str


class SQLiteCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json, created_at FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value_json, created_at = row
        return CacheRecord(key=key, value=json.loads(value_json), created_at=created_at).value

    def set(self, key: str, value: Any) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries(key, value_json, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at
                """,
                (key, json.dumps(value, ensure_ascii=True, sort_keys=True), created_at),
            )

