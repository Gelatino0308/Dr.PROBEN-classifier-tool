"""
Encode breast cancer dataset rows to PROBEN1 format.

Converts 9 raw features (1-10 scale) to normalized values:
- Missing values ('?') → 0.35
- Valid values → value/10
- Binary classification: Benign (2) → [1,0], Malignant → [0,1]
"""
from __future__ import annotations

import argparse
import sys
from typing import List


def _fmt(v: float) -> str:
    # Compact float formatting like %g
    return f"{v:g}"


def encode_row(fields: List[str]) -> str:
    # Expect at least ID + 9 features + class => 11 tokens
    if len(fields) < 11:
        raise ValueError(f"Expected at least 11 comma-separated values, got {len(fields)}: {fields}")

    F = [f.strip() for f in fields]

    out: List[str] = []
    # Features: indices 1..len(F)-2 (skip index 0, skip last)
    for i in range(1, len(F) - 1):
        v = F[i]
        if v == "?":
            out.append("0.35")
        else:
            try:
                out.append(_fmt(float(v) / 10.0))
            except Exception:
                out.append("0.35")

    # Label is last field
    lab = F[-1]
    try:
        lab_i = int(float(lab))
    except Exception:
        lab_i = 4
    # 2 -> Benign => "1 0" ; else -> Malignant => "0 1"
    out.append("1 0" if lab_i == 2 else "0 1")

    return " ".join(out)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode cancer raw to .cod (Python port of raw2cod)")
    parser.add_argument("input", help="Path to breast-cancer-wisconsin.data (use - for stdin)")
    args = parser.parse_args(argv)

    if args.input == "-":
        fin = sys.stdin
        close_fin = False
    else:
        fin = open(args.input, "r", encoding="utf-8")
        close_fin = True

    try:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            fields = s.split(",")
            try:
                encoded = encode_row(fields)
            except Exception as e:
                print(f"[WARN] Skipping malformed line: {s[:120]}... ({e})", file=sys.stderr)
                continue
            print(encoded)
    finally:
        if close_fin:
            fin.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
