import os
import hashlib
import secrets
import sqlite3
from typing import Optional, Dict, List


def get_db_path() -> str:
    return os.environ.get(
        "SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite"
    )


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + "$" + key.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split("$", 1)
    salt = bytes.fromhex(salt_hex)
    expected_key = bytes.fromhex(key_hex)
    new_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return secrets.compare_digest(new_key, expected_key)


def create_user(username: str, password: str) -> Optional[int]:
    username = username.strip()
    if not username or len(username) < 3 or len(username) > 50:
        return None
    if not password or len(password) < 6 or len(password) > 128:
        return None

    conn = get_db_connection()
    try:
        ensure_users_table(conn)
        password_hash = hash_password(password)
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    username = username.strip()
    if not username or not password:
        return None

    conn = get_db_connection()
    try:
        ensure_users_table(conn)
        cursor = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = get_db_connection()
    try:
        ensure_users_table(conn)
        cursor = conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


# ── Conversation management ────────────────────────────────────────


def ensure_conversations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL DEFAULT '新对话',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)"
    )
    conn.commit()


def create_conversation(user_id: int, session_id: str, title: str = "新对话") -> int:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, session_id, title) VALUES (?, ?, ?)",
            (user_id, session_id, title),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_conversations(user_id: int) -> List[Dict]:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        cursor = conn.execute(
            "SELECT id, session_id, title, created_at, updated_at "
            "FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_conversation(session_id: str) -> Optional[Dict]:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        cursor = conn.execute(
            "SELECT id, user_id, session_id, title, created_at, updated_at "
            "FROM conversations WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_conversation_title(session_id: str, title: str) -> None:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE session_id = ?",
            (title, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_conversation_time(session_id: str) -> None:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(session_id: str) -> bool:
    conn = get_db_connection()
    try:
        ensure_conversations_table(conn)
        cursor = conn.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
