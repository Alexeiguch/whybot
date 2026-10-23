"""Semantic table notes (whybot/catalog_notes.json) given to the Planner and SQL agent.

A recording stores the notes it ran with in fixtures/recorded/catalog.json (table_notes), and a replay uses those,
so editing the notes later does not change the replayed prompts."""
import json
from pathlib import Path

NOTES_PATH = Path(__file__).resolve().parent / "catalog_notes.json"


def load_notes() -> dict:
    return {k: v for k, v in json.loads(NOTES_PATH.read_text()).items() if not k.startswith("_")} if NOTES_PATH.exists() else {}
