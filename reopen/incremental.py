"""Kill-tolerant incremental output for eval/analysis sweeps.

WHY (2026-07-17): background tasks get killed by the harness on a
lifetime cap. Write-at-end scripts lose everything (the vocab per-word
sweep lost 14 checkpoints of GPU work; the original per-word record was
destroyed by an unlink-on-start); non-atomic writes can leave a
truncated file that a naive skip-completed check would treat as done —
silent corruption. Rules implemented here:

1. ATOMIC file replacement (tmp + os.replace) — a kill mid-write leaves
   either the old file or the new one, never a partial.
2. RESUME by key — existing valid rows/lines are loaded, their keys
   reported as done, and only missing work is redone.
3. A JSONL file's trailing partial line (kill mid-append) is detected
   and truncated away at open time, so later appends can't create a
   corrupt-middle file.
4. Nothing is ever silently deleted: `archive()` renames to a
   timestamped .bak.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd


def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def archive(path: Path) -> Path | None:
    """Timestamped .bak rename (never delete). Returns the new path."""
    if not path.exists():
        return None
    dest = path.with_suffix(
        path.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
    path.rename(dest)
    return dest


class IncrementalCSV:
    """CSV rebuilt atomically after every append; resumes by key column."""

    def __init__(self, path: Path, key: str = "step"):
        self.path = Path(path)
        self.key = key
        self.rows: list[dict] = []
        if self.path.exists():
            self.rows = pd.read_csv(self.path).to_dict("records")

    @property
    def done(self) -> set:
        return {int(r[self.key]) for r in self.rows}

    def append(self, rows: dict | list[dict]) -> None:
        self.rows.extend([rows] if isinstance(rows, dict) else rows)
        df = pd.DataFrame(self.rows).sort_values(self.key)
        atomic_write_text(self.path, df.to_csv(index=False))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class IncrementalJSONL:
    """Append-per-record JSONL; resumes by key field. On open, a trailing
    partial line (kill mid-append) is truncated away atomically."""

    def __init__(self, path: Path, key: str = "step"):
        self.path = Path(path)
        self.key = key
        self.records: list[dict] = []
        if self.path.exists():
            raw = self.path.read_text(encoding="utf-8").splitlines()
            valid = []
            for line in raw:
                if not line.strip():
                    continue
                try:
                    valid.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # partial tail — drop it and everything after
            self.records = valid
            if len(valid) != len([l for l in raw if l.strip()]):
                atomic_write_text(
                    self.path,
                    "".join(json.dumps(r) + "\n" for r in valid))

    @property
    def done(self) -> set:
        return {int(r[self.key]) for r in self.records}

    def append(self, record: dict) -> None:
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_json_valid(path: Path) -> dict | None:
    """json.load that returns None for missing OR corrupt files, so
    skip-completed checks can't mistake a truncated file for done work."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
