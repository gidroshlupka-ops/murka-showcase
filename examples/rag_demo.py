"""Minimal RAG walkthrough: store a turn + a fact, then recall."""
from __future__ import annotations

import asyncio

from rag_memory import RagMemory


async def main() -> None:
    rag = RagMemory()
    uid = "demo-user"
    probe = await rag.self_test()
    print(probe)

    await rag.add_turn(
        None,
        uid,
        "I moved to Lisbon last spring and started a small bakery.",
        "Nice — Lisbon mornings must smell like bread.",
    )
    await rag.add_fact(None, uid, "Lives in Lisbon. Runs a bakery.")

    hits = await rag.query(None, uid, "where does this person live?", k=3)
    facts = await rag.query_facts(None, uid, "city job", k=3)
    print("turns:", hits)
    print("facts:", facts)


if __name__ == "__main__":
    asyncio.run(main())
