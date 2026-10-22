"""Re-fetch recorded query fixtures from live Zetaris and rewrite rows whose JSON form lost a type.

Fixtures recorded before zetaris_jdbc.to_python stored java.lang.Boolean as 1/0, while the live
prompt (and so the cassette's prompt_sha) rendered it as True/False. Only rows whose values differ
purely by that boxing are rewritten; any real data difference aborts that fixture.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATA_SOURCE"] = "zetaris"
os.environ.pop("SAMPLE_MODE", None)
os.environ.pop("RECORD_FIXTURES", None)

from whybot.config import Settings  # noqa: E402
from whybot.datasource.factory import make_datasource  # noqa: E402

QDIR = Path(__file__).resolve().parents[1] / "fixtures" / "recorded" / "queries"


def same_value(old, new) -> bool:
    if isinstance(new, bool):
        return old in (0, 1) and bool(old) == new
    return old == new or (isinstance(old, float) and isinstance(new, float) and abs(old - new) < 1e-9)


def main() -> None:
    ds = make_datasource(Settings.from_env())
    for f in sorted(QDIR.glob("*.json")):
        fx = json.loads(f.read_text())
        if fx.get("error") or "rows" not in fx:
            continue
        res = ds.run(fx["sql"], timeout_s=180)
        live = json.loads(json.dumps(res.rows))
        if res.columns != fx["columns"] or len(live) != len(fx["rows"]):
            print(f"{f.name}: shape differs, left alone")
            continue
        if not all(same_value(o, n) for ro, rn in zip(fx["rows"], live) for o, n in zip(ro, rn)):
            print(f"{f.name}: values differ, left alone")
            continue
        if json.dumps(live) != json.dumps(fx["rows"]):
            fx["rows"] = live
            f.write_text(json.dumps(fx, indent=1, ensure_ascii=False))
            print(f"{f.name}: rewritten")
        else:
            print(f"{f.name}: unchanged")


if __name__ == "__main__":
    main()
