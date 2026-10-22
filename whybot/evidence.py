from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Evidence:
    qid: str
    hypothesis_id: str
    purpose: str
    iteration: int
    sql: str = ""
    sql_hash: str = ""
    tables: list[str] = field(default_factory=list)
    status: str = "pending"          # ok | failed | skipped
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    elapsed_s: float = 0.0
    source_label: str = ""
    recorded_at: str | None = None
    error: str = ""
    attempts: list[dict] = field(default_factory=list)   # every SQL tried, with error, for repair lessons
    wm_fact_id: str | None = None
    truncated: bool = False

    @property
    def primary_table(self) -> str:
        return self.tables[0] if self.tables else ""

    def to_dict(self) -> dict:
        return asdict(self)

    def for_prompt(self, max_rows: int = 40) -> str:
        if self.status != "ok":
            return f"[{self.qid}] ({self.hypothesis_id}) {self.purpose}\nSTATUS: {self.status} {self.error[:200]}"
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(str(v) for v in r) for r in self.rows[:max_rows])
        more = f"\n... {self.row_count - max_rows} more rows not shown" if self.row_count > max_rows else ""
        if self.truncated:
            more += "\nWARNING: the result was truncated at the row cap; treat totals as partial"
        return (f"[{self.qid}] ({self.hypothesis_id}) {self.purpose}\nsql_hash={self.sql_hash} tables={','.join(self.tables)}\n"
                f"{head}\n{body}{more}")
