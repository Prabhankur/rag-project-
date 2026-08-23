"""
Persistent paper library.
- Every processed paper's hierarchical tree is saved to disk permanently
  (trees/<id>.json), keyed by a hash of its title.
- Re-uploading a paper with the same title is detected and skipped -
  no re-parsing, the cached tree is loaded instantly.
- This is what makes multi-paper / cross-paper chat possible: papers
  persist across app restarts, so you can build up a library over time.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "library.db"
TREES_DIR = BASE_DIR / "trees"
TREES_DIR.mkdir(exist_ok=True)


def init_library_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def paper_id_from_title(title: str) -> str:
    """Deterministic ID from the paper's title - same title always
    resolves to the same ID, which is how we detect duplicates."""
    normalized = title.strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def get_paper_by_title(title: str):
    """Returns {"id","title","filename","created_at","tree"} if this
    exact title is already in the library, else None."""
    pid = paper_id_from_title(title)
    tree_path = TREES_DIR / f"{pid}.json"
    if not tree_path.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, filename, created_at FROM papers WHERE id = ?", (pid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None

    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    return {"id": row[0], "title": row[1], "filename": row[2], "created_at": row[3], "tree": tree}


def save_paper(title: str, filename: str, tree: dict) -> str:
    pid = paper_id_from_title(title)
    tree_path = TREES_DIR / f"{pid}.json"
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree, f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO papers (id, title, filename, created_at) VALUES (?, ?, ?, ?)",
        (pid, title, filename, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return pid


def list_papers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, filename, created_at FROM papers ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "filename": r[2], "created_at": r[3]} for r in rows]


def load_tree(paper_id: str):
    tree_path = TREES_DIR / f"{paper_id}.json"
    if not tree_path.exists():
        return None
    with open(tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_paper(paper_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()
    tree_path = TREES_DIR / f"{paper_id}.json"
    if tree_path.exists():
        tree_path.unlink()