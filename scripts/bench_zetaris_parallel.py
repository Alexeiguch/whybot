"""Time the same queries against live Zetaris one after another and then side by side on the session pool.

Usage: .venv/bin/python scripts/bench_zetaris_parallel.py sqls.json [workers]
Prints timings only (never rows or credentials)."""
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whybot.config import Settings  # noqa: E402
from whybot.datasource.zetaris_jdbc import ZetarisJDBCSource  # noqa: E402

s = Settings.from_env()
sqls = json.loads(Path(sys.argv[1]).read_text())
workers = int(sys.argv[2]) if len(sys.argv) > 2 else len(sqls)
t = time.time()
ds = ZetarisJDBCSource(s.zetaris_databases, s.zetaris_tables)
print(f"connect {time.time() - t:.1f}s")
t = time.time()
ds.catalog()
print(f"catalog (parallel describe of {len(s.zetaris_tables)} tables) {time.time() - t:.1f}s")


def one(q):
    t0 = time.time()
    r = ds.run(q, timeout_s=300)
    return round(time.time() - t0, 1), len(r.rows)


t = time.time()
ser = [one(q) for q in sqls]
print(f"serial   total {time.time() - t:.1f}s  per query {ser}")
t = time.time()
with cf.ThreadPoolExecutor(max_workers=workers) as ex:
    par = list(ex.map(one, sqls))
print(f"parallel total {time.time() - t:.1f}s  per query {par}  (workers={workers}, sessions={len(ds._conns)})")
ds.close()
