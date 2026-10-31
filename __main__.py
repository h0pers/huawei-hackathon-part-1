"""Portable entry point for the isolated research snapshot (Python 3.13, Linux/macOS)."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "models/research_workspace"


def prepare():
    """Restore small metadata and layout aliases, never models or training data."""
    WORK.mkdir(parents=True, exist_ok=True)
    for name, target in {
        "attack": REPO / "techarena/research/attack",
        "huawei-hackathon-part-1-master": REPO,
    }.items():
        alias = WORK / name
        if alias.is_symlink() and alias.resolve() == target:
            continue
        if alias.exists() or alias.is_symlink():
            raise SystemExit(f"Refusing to replace {alias}")
        alias.symlink_to(target, target_is_directory=True)
    output = WORK / "attack_results"
    for path in (REPO / "references/research_metadata").rglob("*.json"):
        destination = output / path.relative_to(REPO / "references/research_metadata")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
    (WORK / "attack_cache").mkdir(exist_ok=True)
    (WORK / "submission_history").mkdir(exist_ok=True)
    return WORK


def verify_submission(path, evaluation, expected_rows=529390, expected_sha256=None):
    """Check all row IDs, order, probabilities, and optional immutable ZIP fingerprint."""
    if expected_sha256:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_sha256:
            raise ValueError("SHA256 does not match retained artifact")
    with evaluation.open(newline="", encoding="utf-8") as stream:
        expected = [row["rowid"] for row in csv.DictReader(stream)]
    if len(expected) != expected_rows or len(set(expected)) != expected_rows:
        raise ValueError("Unexpected evaluation row count or duplicate rowids")

    def check(stream):
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["rowid", "predicted_probability"]:
            raise ValueError("Wrong submission columns")
        ids = []
        for row in reader:
            score = float(row["predicted_probability"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid probability")
            ids.append(row["rowid"])
        if ids != expected:
            raise ValueError("Row IDs, count or order differ from evaluation")
        return {"rows": len(ids), "unique_rowids": len(set(ids)), "valid": True}

    if path.suffix == ".zip":
        import io

        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC failed")
            names = archive.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                raise ValueError("ZIP must contain exactly one CSV")
            with archive.open(names[0]) as data:
                return check(io.TextIOWrapper(data, encoding="utf-8", newline=""))
    with path.open(newline="", encoding="utf-8") as stream:
        return check(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prepare")
    actions.add_parser("test")
    verify = actions.add_parser("verify")
    verify.add_argument("submission", type=Path)
    verify.add_argument("--evaluation", type=Path, default=REPO / "data/raw/evaluation.csv")
    verify.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "verify":
        print(
            json.dumps(
                verify_submission(args.submission, args.evaluation, expected_sha256=args.sha256),
                indent=2,
            )
        )
        return
    workspace = prepare()
    if args.action == "prepare":
        print(workspace)
        return
    env = os.environ.copy()
    env.update(
        TECHARENA_RESEARCH_ROOT=str(workspace),
        TECHARENA_DATA_DIR=str(REPO / "data/interim"),
        PYTHONPATH=os.pathsep.join(
            [str(REPO), str(REPO / "techarena/research"), os.environ.get("PYTHONPATH", "")]
        ),
        POLARS_MAX_THREADS="8",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="1",
    )
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests"], cwd=REPO, env=env))


if __name__ == "__main__":
    main()
