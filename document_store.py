"""
Persistent paper library.
Each paper's parsed section tree is pickled to disk, keyed by a slug of
its filename/title. Re-uploading the same paper skips re-parsing entirely -
this is the "Persistent DB" node in the architecture:

    Paper A -> Tree A ─┐
    Paper B -> Tree B ─┼─> Persistent DB (this file's storage)
    Paper C -> Tree C ─┘
"""

import re
import json
import pickle
from pathlib import Path
from datetime import datetime

STORE_DIR = Path(__file__).parent / "paper_store"
STORE_DIR.mkdir(exist_ok=True)
INDEX_PATH = STORE_DIR / "index.json"


def slugify(name: str) -> str:
    """Turns a filename/title into a stable unique ID. Since research
    paper titles are unique, this doubles as the paper's permanent ID."""
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip().lower()).strip("-")
    return slug or "paper"


def _load_index() -> dict:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return {}


def _save_index(index: dict):
    INDEX_PATH.write_text(json.dumps(index, indent=2))


def paper_exists(paper_id: str) -> bool:
    return (STORE_DIR / f"{paper_id}.pkl").exists()


def save_paper(paper_id: str, title: str, tree: dict):
    with open(STORE_DIR / f"{paper_id}.pkl", "wb") as f:
        pickle.dump(tree, f)
    index = _load_index()
    index[paper_id] = {"title": title, "added_at": datetime.now().isoformat()}
    _save_index(index)


def load_paper(paper_id: str) -> dict:
    with open(STORE_DIR / f"{paper_id}.pkl", "rb") as f:
        return pickle.load(f)


def list_papers() -> list:
    index = _load_index()
    return [{"id": pid, **meta} for pid, meta in index.items()]


def delete_paper(paper_id: str):
    p = STORE_DIR / f"{paper_id}.pkl"
    if p.exists():
        p.unlink()
    index = _load_index()
    index.pop(paper_id, None)
    _save_index(index)