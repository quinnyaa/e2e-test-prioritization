"""Stage 8 (8A): verify the files frozen before the hold-out test part is opened.

`ml/stage8_frozen.json` holds
  frozen       the sha256 table of notes/stage8-protocol.md, p. 1 (45 files, transferred verbatim);
  added_in_8a  files added to the lock in step 8A (Kim-Porter settings, evaluation code; protocol, section 9).

Checks: (1) `frozen` equals the table parsed from the protocol (path set and hashes, exactly 45 rows);
(2) `added_in_8a` equals the `path` `sha256` pairs recorded in section 9 of the protocol;
(3) every listed file exists and has the recorded sha256. Exit code 1 on any mismatch.
ml/run_stage8.py calls `verify()` before every run and refuses to evaluate the test part if it fails.

Usage:  ml/.venv/bin/python ml/check_frozen.py [-q]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_JSON = ROOT / "ml" / "stage8_frozen.json"
PROTOCOL = ROOT / "notes" / "stage8-protocol.md"
N_PROTOCOL_FILES = 45
ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", re.M)
LOG_PAIR = re.compile(r"`([^`\s]+)` `([0-9a-f]{64})`")


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_table() -> dict[str, str]:
    """sha256 rows of section 1 of the protocol (path in backticks, 64 hex digits)."""
    text = PROTOCOL.read_text(encoding="utf-8")
    section = text.split("## 1. Що заморожено", 1)[1].split("\n## 2.", 1)[0]
    return dict(ROW.findall(section))


def protocol_log_hashes() -> dict[str, str]:
    """`path` `sha256` pairs recorded in section 9 (journal of deviations) of the protocol."""
    text = PROTOCOL.read_text(encoding="utf-8")
    log = text.split("## 9. Журнал відхилень", 1)[1]
    return dict(LOG_PAIR.findall(log))


def load_frozen() -> dict:
    return json.loads(FROZEN_JSON.read_text(encoding="utf-8"))


def verify(verbose: bool = True) -> list[str]:
    """List of problems (empty = every frozen file matches)."""
    problems: list[str] = []
    data = load_frozen()
    frozen, added = data["frozen"], data.get("added_in_8a", {})
    table = protocol_table()
    if len(table) != N_PROTOCOL_FILES:
        problems.append(f"protocol table has {len(table)} rows, expected {N_PROTOCOL_FILES}")
    if frozen != table:
        diff = sorted(set(frozen.items()) ^ set(table.items()))
        problems.append(f"stage8_frozen.json `frozen` differs from the protocol table: {diff[:4]}")
    logged = protocol_log_hashes()
    if added != logged:
        diff = sorted(set(added.items()) ^ set(logged.items()))
        problems.append(f"stage8_frozen.json `added_in_8a` differs from the sha256 recorded in protocol section 9: {diff[:4]}")
    overlap = set(frozen) & set(added)
    if overlap:
        problems.append(f"files listed both as frozen and as added in 8A: {sorted(overlap)}")
    for section, entries in (("frozen", frozen), ("added_in_8a", added)):
        ok = 0
        for rel, expected in entries.items():
            path = ROOT / rel
            if not path.is_file():
                problems.append(f"{section}: {rel} is missing")
                continue
            got = sha256(path)
            if got != expected:
                problems.append(f"{section}: {rel} sha256 {got[:12]}… != recorded {expected[:12]}…")
            else:
                ok += 1
        if verbose:
            print(f"  {section}: {ok} of {len(entries)} files match")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    problems = verify(verbose=not args.quiet)
    for p in problems:
        print(f"  [FAIL] {p}")
    print("frozen files: " + ("all match" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
