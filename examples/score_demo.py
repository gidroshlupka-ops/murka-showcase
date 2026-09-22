"""Print hybrid scores from the real `_score` function — no fake UI."""
from __future__ import annotations

import time

from rag_memory import _score

now = time.time()
# Distances are cosine-from-Chroma style (0 = identical). Ages in hours via timestamps.
# Off-topic recent chatter vs a slightly older but on-topic high-importance fact.
rows = [
    ("turn  'lol wait what'", 0.48, now - 12 * 60, 0.45),
    ("turn  morning bread joke", 0.36, now - 3 * 3600, 0.45),
    ("fact  Lives in Lisbon. Runs a bakery.", 0.14, now - 40 * 3600, 0.90),
]
print("score = 0.5*sim + 0.3*exp(-age_h/72) + 0.2*imp")
print(f"{'doc':40} {'dist':>6} {'sim':>6} {'age_h':>7} {'imp':>5} {'score':>7}")
ranked = []
for doc, dist, ts, imp in rows:
    sim = max(0.0, 1.0 - dist)
    age_h = (now - ts) / 3600
    s = _score(dist, ts, imp)
    ranked.append((s, doc, dist, sim, age_h, imp))
    print(f"{doc:40} {dist:6.2f} {sim:6.2f} {age_h:7.1f} {imp:5.2f} {s:7.3f}")
print()
print("rank after hybrid sort:")
for i, (s, doc, *_rest) in enumerate(sorted(ranked, reverse=True), 1):
    print(f"  {i}. {s:.3f}  {doc}")
