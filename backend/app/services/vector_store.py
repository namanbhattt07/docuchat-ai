from pathlib import Path
from threading import Lock

import chromadb

from app.core.config import get_settings

_collection = None
_lock = Lock()


def get_collection():
    """Reuse a single persistent Chroma client/collection for the process
    lifetime. Opening a fresh `PersistentClient` on every request re-read the
    on-disk HNSW metadata from scratch each time -- wasteful at any scale and
    a real bottleneck under concurrent requests.
    """
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                settings = get_settings()
                path = Path(settings.chroma_persist_directory)
                path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(path))
                _collection = client.get_or_create_collection(name="docuchat_chunks", metadata={"hnsw:space": "cosine"})
    return _collection
