"""Standalone submission check against the raw evaluation.csv. Usage: validate_submission.py FILE [REFERENCE]"""
import csv, math, sys
import numpy as np
from scipy.stats import spearmanr
path = sys.argv[1]
csv.field_size_limit(10**9)
with open("data/raw/evaluation.csv", newline="", encoding="utf-8") as f:
    expected = {row["rowid"] for row in csv.DictReader(f)}
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
errors = []
if header != ["rowid", "predicted_probability"]:
    errors.append(f"bad header {header}")
if any(len(r) != 2 for r in rows):
    errors.append("a line does not have exactly 2 columns")
ids = [r[0] for r in rows]
seen = set(ids)
if len(rows) != 529_390:
    errors.append(f"{len(rows)} rows, expected 529390")
if len(expected) != 529_390:
    errors.append(f"evaluation.csv has {len(expected)} unique rowids")
if len(seen) != len(ids):
    errors.append(f"{len(ids) - len(seen)} duplicate rowids")
if seen - expected:
    errors.append(f"{len(seen - expected)} unknown rowids")
if expected - seen:
    errors.append(f"{len(expected - seen)} missing rowids")
values = {}
for r in rows:
    try:
        v = float(r[1])
    except ValueError:
        errors.append(f"not a number: {r[1]!r}")
        break
    if not math.isfinite(v) or not 0 <= v <= 1:
        errors.append(f"out of range: {r[1]!r}")
        break
    values[r[0]] = v
if errors:
    print("INVALID:", "; ".join(errors))
    sys.exit(1)
a = np.fromiter(values.values(), float)
print(f"VALID {path}: {len(rows)} rows, min {a.min():.6f} mean {a.mean():.6f} max {a.max():.6f}, distinct {len(np.unique(a))}")
if len(sys.argv) > 2:
    with open(sys.argv[2], newline="", encoding="utf-8") as f:
        ref = {r["rowid"]: float(r["predicted_probability"]) for r in csv.DictReader(f)}
    b = np.array([ref[k] for k in values])
    print(f"Spearman with {sys.argv[2]}: {spearmanr(a, b).statistic:.5f}")
