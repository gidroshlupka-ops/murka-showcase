"""Key rotation: RPM vs daily-quota vs group soft-limit."""
from __future__ import annotations

import tempfile
from pathlib import Path

from key_manager import KeyManager

FAKE = [
    "aaaaaaaa" + "x" * 32,  # group A
    "aaaaaaaa" + "y" * 32,  # group A (same project)
    "bbbbbbbb" + "z" * 32,  # group B
]


def main() -> None:
    db = Path(tempfile.gettempdir()) / "murka_showcase_key_bans.db"
    km = KeyManager(FAKE, db_path=str(db))
    idx, key = km.pick_best("chat")
    print("picked", idx, key[:8] + "…")
    km.mark_used(idx)

    km.ban_429(idx, err_body="RESOURCE_EXHAUSTED: quota exceeded per_day")
    idx2, key2 = km.pick_best("chat")
    print("after RPD, next", idx2, (key2[:8] + "…") if key2 else "none")
    print("all daily-banned?", km.all_rpd_banned())


if __name__ == "__main__":
    main()
