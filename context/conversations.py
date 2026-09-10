import logging
import sqlite3
from pathlib import Path

from context.entities import Message

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, conversations_db_path: Path | None = None) -> None:
        if conversations_db_path is None:
            conversations_db_path = Path("data") / "dbs" / "conversations.db"
        conversations_db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(conversations_db_path)

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        self.conn.execute("PRAGMA foreign_keys = ON;")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages  (
                id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def create_conversation(self, title: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (title, created_at, updated_at)
            VALUES (?, datetime('now'), datetime('now'))
            """,
            (title,),
        )
        self.conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to create conversation")
        logger.info("Created conversation (id=%d)", cursor.lastrowid)
        return cursor.lastrowid

    def get_conversations(self, limit: int | None = None) -> list[tuple[int, str]]:
        cursor = self.conn.cursor()

        if limit is None:
            cursor.execute(
                "SELECT id, title FROM conversations ORDER BY updated_at DESC"
            )
        elif limit <= 0:
            raise ValueError("Limit must be a positive integer")
        else:
            cursor.execute(
                """
                SELECT id, title
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    def delete_conversation(self, conversation_id: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        self.conn.commit()
        logger.info("Deleted conversation (id=%d)", conversation_id)

    def conversation_exists(self, conversation_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.fetchone() is not None

    def rename_conversation(self, conversation_id: int, new_title: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (new_title, conversation_id),
        )
        self.conn.commit()
        logger.info("Renamed conversation (id=%d)", conversation_id)

    def add_message(self, conversation_id: int, msg: Message) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (conversation_id, msg.role, msg.content),
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,)
        )
        self.conn.commit()
        logger.info(
            "Persisted conversation message (conversation_id=%d, role=%s)",
            conversation_id,
            msg.role,
        )


    def get_messages(
        self,
        conversation_id: int,
        limit: int | None = None,
    ) -> list[Message]:
        cursor = self.conn.cursor()

        if limit is None:
            cursor.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            )
            rows = cursor.fetchall()
        elif limit <= 0:
            raise ValueError("Limit must be a positive integer")
        else:
            cursor.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = list(reversed(cursor.fetchall()))
        return [Message(role=row[0], content=row[1]) for row in rows]
