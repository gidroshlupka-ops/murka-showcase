"""
LLM key pool with rate-limit-aware rotation.

429 handling
------------
* RPM  → 65s cooldown on that key
* RPD / quota exceeded → 24h cooldown on the whole account group
* after 3 retries (e.g. different proxies) → 12h on that key only
* provider `limit: 0` (fresh unused key) → 1h rest
* revoked / suspended → ~permanent ban

Groups
------
Keys sharing a prefix (first 8 chars) are treated as one billing project.
A daily soft cap (~900 calls, reset at 08:00 UTC) spreads load so one spike
does not burn the entire free-tier quota.

Bans persist in SQLite so a process restart does not revive dead keys.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import time

from logging_setup import log


def load_pool_from_env(prefix: str = "LLM_KEY_") -> list[str]:
    keys: list[str] = []
    for i in range(1, 201):
        k = os.environ.get(f"{prefix}{i}", "").strip()
        if k:
            keys.append(k)
    return keys


class KeyManager:
    COOLDOWN_RPM = 65
    COOLDOWN_RPD = 86400
    COOLDOWN_RPD_AFTER_RETRIES = 43200
    COOLDOWN_LIMIT_ZERO = 3600
    COOLDOWN_PERMANENT = 10 * 365 * 86400
    _GRP_DAILY_SOFT_LIMIT = 900

    def __init__(self, pool: list[str], *, db_path: str | None = None):
        self._pool = [k for k in pool if k and len(k) > 20]
        self._idx = 0
        self._cooldown: dict[int, float] = {}
        self._last_used: dict[int, float] = {}
        self._err_count: dict[int, int] = {}
        self._type_idx: dict[str, int] = {"chat": 0, "vision": 0, "transcribe": 0}
        self._groups: dict[str, list[int]] = {}
        self._grp_day_count: dict[str, int] = {}
        self._grp_day_reset: dict[str, float] = {}
        default_db = "/data/key_bans.db" if os.path.isdir("/data") else "key_bans.db"
        self._ban_db = db_path or default_db

        for i, k in enumerate(self._pool):
            grp = k[:8]
            self._groups.setdefault(grp, []).append(i)
        if len(self._groups) < len(self._pool):
            log.info(
                "KeyManager: %d keys in %d groups",
                len(self._pool),
                len(self._groups),
            )
        self._init_ban_db()
        self._load_bans()
        if not self._pool:
            log.warning("key pool is empty")
        else:
            active = sum(1 for i in range(len(self._pool)) if not self._is_banned(i))
            log.info("KeyManager: %d keys, %d active", len(self._pool), active)

    def _init_ban_db(self):
        with sqlite3.connect(self._ban_db) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS bans(
                idx INTEGER PRIMARY KEY,
                until_ts REAL NOT NULL)"""
            )

    def _load_bans(self):
        now = time.monotonic()
        wall_now = time.time()
        try:
            with sqlite3.connect(self._ban_db) as c:
                rows = c.execute("SELECT idx, until_ts FROM bans").fetchall()
            loaded, expired_idxs = 0, []
            for idx, until_wall in rows:
                remaining = until_wall - wall_now
                if remaining > 0:
                    self._cooldown[idx] = now + remaining
                    loaded += 1
                else:
                    expired_idxs.append(idx)
            if expired_idxs:
                with sqlite3.connect(self._ban_db) as c:
                    c.executemany(
                        "DELETE FROM bans WHERE idx=?",
                        [(i,) for i in expired_idxs],
                    )
            log.info("KeyManager: loaded %d active bans", loaded)
        except Exception as e:
            log.warning("KeyManager: could not load bans: %s", e)

    def _save_ban(self, idx: int, duration: float):
        until_wall = time.time() + duration
        try:
            with sqlite3.connect(self._ban_db) as c:
                existing = c.execute(
                    "SELECT until_ts FROM bans WHERE idx=?", (idx,)
                ).fetchone()
                if existing and existing[0] > until_wall:
                    log.info("KeyManager: ban #%d already longer, skip", idx)
                    return
                c.execute(
                    "INSERT OR REPLACE INTO bans(idx, until_ts) VALUES(?,?)",
                    (idx, until_wall),
                )
        except Exception as e:
            log.warning("KeyManager: could not persist ban: %s", e)

    def _is_banned(self, idx: int) -> bool:
        return time.monotonic() < self._cooldown.get(idx, 0)

    def is_permanent(self, idx: int) -> bool:
        return (self._cooldown.get(idx, 0) - time.monotonic()) > (365 * 86400)

    def ban_permanent(self, idx: int, err_body: str = ""):
        cd = float(self.COOLDOWN_PERMANENT)
        self._cooldown[idx] = time.monotonic() + cd
        self._save_ban(idx, cd)
        log.warning("key #%d permanently banned: %s", idx, (err_body or "")[:180])

    def mark_limit_zero(self, idx: int):
        """Provider returned limit: 0 — park the unused key for 1h."""
        cd = float(self.COOLDOWN_LIMIT_ZERO)
        self._cooldown[idx] = time.monotonic() + cd
        self._save_ban(idx, cd)
        log.info("key #%d → limit 0, rest 1h", idx)

    def ban_429(self, idx: int, err_body: str = "", *, after_retries: bool = False):
        if after_retries:
            cd = float(self.COOLDOWN_RPD_AFTER_RETRIES)
            self._cooldown[idx] = time.monotonic() + cd
            self._save_ban(idx, cd)
            log.warning("key #%d → 429 after retries, ban 12h", idx)
            return
        body_l = err_body.lower()
        is_rpd = (
            "free_tier_requests" in body_l
            or "per_day" in body_l
            or "requests_per_day" in body_l
            or ("quota exceeded" in body_l and "free_tier" in body_l)
            or "daily" in body_l
            or "quota_exceeded" in body_l
            or ("quota" in body_l and "exceeded" in body_l)
            or "resource_exhausted" in body_l
            or "rate_limit_exceeded" in body_l
        )
        if is_rpd:
            cd = float(self.COOLDOWN_RPD)
            grp_key = self._pool[idx][:8] if idx < len(self._pool) else ""
            group_idxs = self._groups.get(grp_key, [idx])
            if len(group_idxs) > 1:
                log.warning(
                    "key #%d → RPD, ban group %s (%d keys) for 24h",
                    idx,
                    grp_key,
                    len(group_idxs),
                )
            else:
                log.warning("key #%d → daily quota, ban 24h", idx)
            new_cd_end = time.monotonic() + cd
            for gidx in group_idxs:
                if self._cooldown.get(gidx, 0) < new_cd_end:
                    self._cooldown[gidx] = new_cd_end
                    self._save_ban(gidx, cd)
        else:
            cd = float(self.COOLDOWN_RPM)
            log.info("key #%d → RPM 429, ban 65s", idx)
            new_cd_end = time.monotonic() + cd
            if self._cooldown.get(idx, 0) >= new_cd_end:
                return
            self._cooldown[idx] = new_cd_end
            self._save_ban(idx, cd)

    def mark_used(self, idx: int):
        self._last_used[idx] = time.monotonic()
        self._err_count[idx] = 0
        if idx < len(self._pool):
            self._grp_inc(self._pool[idx][:8])

    def mark_error(self, idx: int):
        self._err_count[idx] = self._err_count.get(idx, 0) + 1
        if self._err_count[idx] >= 3:
            self._cooldown[idx] = time.monotonic() + 10.0
            log.warning("key #%d → soft ban 10s (3 errors)", idx)
            self._err_count[idx] = 0

    def _grp_soft_limited(self, grp_prefix: str) -> bool:
        now = time.time()
        reset_ts = self._grp_day_reset.get(grp_prefix, 0)
        if now >= reset_ts:
            now_utc = datetime.datetime.now(datetime.UTC)
            tomorrow = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
            if now_utc.hour >= 8:
                tomorrow = tomorrow + datetime.timedelta(days=1)
            self._grp_day_reset[grp_prefix] = tomorrow.timestamp()
            self._grp_day_count[grp_prefix] = 0
        return self._grp_day_count.get(grp_prefix, 0) >= self._GRP_DAILY_SOFT_LIMIT

    def _grp_inc(self, grp_prefix: str):
        self._grp_day_count[grp_prefix] = self._grp_day_count.get(grp_prefix, 0) + 1

    def pick_best(self, req_type: str = "chat") -> tuple[int, str]:
        if not self._pool:
            return -1, ""
        n = len(self._pool)
        start = self._type_idx.get(req_type, 0) % n

        for pass_num in range(2):
            for offset in range(n):
                candidate = (start + offset) % n
                if self._is_banned(candidate):
                    continue
                grp = self._pool[candidate][:8]
                if pass_num == 0 and self._grp_soft_limited(grp):
                    continue
                self._type_idx[req_type] = candidate
                return candidate, self._pool[candidate]
        return -1, ""

    def advance(self, req_type: str = "chat"):
        n = len(self._pool)
        if n:
            cur = self._type_idx.get(req_type, 0)
            self._type_idx[req_type] = (cur + 1) % n

    def all_banned(self) -> bool:
        return all(self._is_banned(i) for i in range(len(self._pool)))

    def all_rpd_banned(self) -> bool:
        now = time.monotonic()
        if not self._pool:
            return True
        return all(self._cooldown.get(i, 0) - now > 3600 for i in range(len(self._pool)))

    def next_cooldown_end(self) -> float:
        now = time.monotonic()
        times = [self._cooldown.get(i, 0) for i in range(len(self._pool))]
        return min((t for t in times if t > now), default=0)

    def next_available(self, req_type: str = "chat") -> str:
        _, key = self.pick_best(req_type)
        return key

    def __len__(self):
        return len(self._pool)
