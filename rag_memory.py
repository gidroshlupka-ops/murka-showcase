"""
Semantic long-term memory (RAG) with a hybrid ranker.

Each user is isolated by `uid` in ChromaDB metadata — conversations never mix.
Retrieval is not "nearest vector wins": cosine similarity is mixed with recency
and explicit importance so a week-old fact can still beat a noisy recent turn.

Stores
------
turns     dialogue snippets (what they said / what the assistant said)
facts     durable user facts (name, job, preferences) — higher importance
sessions  short human notes when someone disappears for hours
"""
from __future__ import annotations

import asyncio
import math
import os
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor

import aiohttp

from logging_setup import log

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-embed")
_st_model = None
_st_lock = __import__("threading").Lock()


def _load_model():
    global _st_model
    with _st_lock:
        if _st_model is not None:
            return _st_model
        from sentence_transformers import SentenceTransformer

        log.info("rag: loading local embedding model %s", EMBED_MODEL)
        _st_model = SentenceTransformer(EMBED_MODEL, device="cpu")
        log.info("rag: model ready")
        return _st_model


def _encode_sync(text: str) -> list[float] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        model = _load_model()
        vec = model.encode(text[:2000], normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        log.warning("rag: encode fail: %s", e)
        return None


async def _embed(_session: aiohttp.ClientSession | None, text: str) -> list[float] | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_embed_pool, _encode_sync, text)


def _score(dist: float, ts: float, importance: float) -> float:
    """0.5 similarity + 0.3 recency (72h half-life) + 0.2 importance."""
    sim = max(0.0, 1.0 - float(dist))
    age_h = max(0.0, (time.time() - float(ts or 0)) / 3600.0)
    recency = math.exp(-age_h / 72.0)
    imp = max(0.0, min(1.0, float(importance or 0.4)))
    return 0.5 * sim + 0.3 * recency + 0.2 * imp


class _Store:
    def __init__(self, name: str, subdir: str):
        self.name = name
        self._enabled = chromadb is not None
        self._col = None
        if not self._enabled:
            return
        path = f"/data/{subdir}" if os.path.isdir("/data") else subdir
        os.makedirs(path, exist_ok=True)
        try:
            client = chromadb.PersistentClient(path=path)
            self._col = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
            )
        except Exception as e:
            log.warning("rag: init %s failed: %s", name, e)
            self._enabled = False

    async def add(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        text: str,
        doc: str,
        *,
        kind: str = "turn",
        importance: float = 0.4,
    ) -> bool:
        if not self._enabled or not self._col:
            return False
        try:
            emb = await _embed(session, text)
            if not emb:
                return False
            self._col.add(
                documents=[doc],
                embeddings=[emb],
                ids=[f"{uid}_{kind}_{_uuid.uuid4().hex}"],
                metadatas=[{
                    "uid": uid,
                    "ts": time.time(),
                    "kind": kind,
                    "importance": float(importance),
                }],
            )
            return True
        except Exception as e:
            log.warning("rag: %s add failed: %s", self.name, e)
            return False

    async def query(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        query_text: str,
        k: int,
    ) -> list[str]:
        if not self._enabled or not self._col or not query_text.strip():
            return []
        try:
            n_uid = self.count_for_uid(uid)
            if n_uid <= 0:
                return []
            emb = await _embed(session, query_text)
            if not emb:
                return []
            take = min(max(k * 4, k), n_uid, 24)
            try:
                res = self._col.query(
                    query_embeddings=[emb],
                    n_results=take,
                    where={"uid": uid},
                    include=["documents", "metadatas", "distances"],
                )
            except TypeError:
                res = self._col.query(
                    query_embeddings=[emb],
                    n_results=take,
                    where={"uid": uid},
                )
        except Exception as e:
            log.warning("rag: %s query failed: %s", self.name, e)
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not dists:
            dists = [0.2] * len(docs)
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            key = doc.strip()[:180]
            if key in seen:
                continue
            seen.add(key)
            meta = meta or {}
            ranked.append((
                _score(dist, meta.get("ts") or 0, meta.get("importance") or 0.4),
                doc,
            ))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:k]]

    def count_for_uid(self, uid: str) -> int:
        if not self._enabled or not self._col:
            return 0
        try:
            got = self._col.get(where={"uid": uid})
            return len(got.get("ids") or [])
        except Exception:
            return 0

    def delete_uid(self, uid: str) -> int:
        if not self._enabled or not self._col:
            return 0
        try:
            n = self.count_for_uid(uid)
            if n:
                self._col.delete(where={"uid": uid})
            return n
        except Exception as e:
            log.warning("rag: delete uid=%s %s: %s", uid, self.name, e)
            return 0


class RagMemory:
    def __init__(self) -> None:
        if chromadb is None:
            log.warning("chromadb is not installed — semantic memory disabled")
        else:
            log.info("rag: chromadb available")
        self._turns = _Store("life_turns", "chroma_turns")
        self._facts = _Store("life_facts", "chroma_facts")
        self._last_snap_at: dict[str, float] = {}

    async def add_turn(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        doc = f"user: {user_text}\nassistant: {assistant_text}"
        ok = await self._turns.add(
            session, uid, doc, doc, kind="turn", importance=0.45,
        )
        if ok:
            log.info("rag: stored turn uid=%s (n=%d)", uid, self._turns.count_for_uid(uid))
        else:
            log.warning("rag: turn not stored uid=%s", uid)

    async def query(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        query_text: str,
        k: int = 5,
    ) -> list[str]:
        hits = await self._turns.query(session, uid, query_text, k)
        log.info("rag: recall uid=%s → %d chunks", uid, len(hits))
        return hits

    async def add_fact(self, session: aiohttp.ClientSession | None, uid: str, fact: str) -> None:
        await self._facts.add(session, uid, fact, fact, kind="fact", importance=0.9)

    async def query_facts(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        query_text: str,
        k: int = 12,
    ) -> list[str]:
        return await self._facts.query(session, uid, query_text, k)

    async def snapshot_session(
        self,
        session: aiohttp.ClientSession | None,
        uid: str,
        away_sec: float,
        user_bits: list[str],
        asst_bits: list[str],
        last_seen: float,
    ) -> None:
        """When a user is gone for 2h+, store a short note about the last visit."""
        if away_sec < 7200 or len(user_bits) < 2:
            return
        if self._last_snap_at.get(uid) == last_seen:
            return
        hours = int(away_sec // 3600)
        doc = (
            f"previous visit (then away ~{hours}h): "
            + " / ".join(user_bits)
            + ". assistant then: "
            + " / ".join(asst_bits)
        )
        ok = await self._turns.add(
            session, uid, doc, doc, kind="session", importance=0.75,
        )
        if ok:
            self._last_snap_at[uid] = last_seen
            log.info("rag: session snapshot uid=%s", uid)

    def count_for_uid(self, uid: str) -> int:
        return self._turns.count_for_uid(uid)

    def facts_count_for_uid(self, uid: str) -> int:
        return self._facts.count_for_uid(uid)

    def forget_uid(self, uid: str) -> tuple[int, int]:
        t = self._turns.delete_uid(uid)
        f = self._facts.delete_uid(uid)
        self._last_snap_at.pop(uid, None)
        log.info("rag: forgot uid=%s turns=%d facts=%d", uid, t, f)
        return t, f

    async def self_test(self, session: aiohttp.ClientSession | None = None) -> str:
        emb = await _embed(session, "probe phrase to verify embeddings")
        if not emb:
            return "embedding model failed (install sentence-transformers)"
        return f"ok — vector dim {len(emb)} ({EMBED_MODEL})"
