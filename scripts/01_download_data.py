#!/usr/bin/env python3
"""
Download the security corpora we train on.

We deliberately use four datasets with *different* shapes, because a model
trained on one narrow kind of text learns one narrow kind of text:

  1. NIST publications  (ethanolivertroy/nist-cybersecurity-training)
     Long, clean, well-edited technical prose. This is what teaches the
     model grammar, long-range structure, and professional register.

  2. CVE records       (CIRCL/vulnerability-scores)
     837K real vulnerabilities with descriptions, CVSS scores and CWE
     classes. This is the domain knowledge: what a vulnerability IS.

  3. CVE Q&A           (rezaduty/cybersecurity-qa-v2)
     2.6M question/answer pairs derived from NVD. This teaches the model
     the *shape* of an answer ("What is CVE-2021-...? It is ..."), which
     is what makes a base model usable as an assistant later.

  4. PreTrain 640k     (11-47/Organized_PreTrain_Cyber_Security_640k)
     Already formatted as a pretraining corpus. Cheap to add, and it
     broadens the topical coverage.

Run:  python scripts/01_download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


# (dataset_id, [(path_in_repo, local_subdir)])
TARGETS = [
    (
        "ethanolivertroy/nist-cybersecurity-training",
        ["train.jsonl", "valid.jsonl"],
        "nist",
    ),
    (
        "CIRCL/vulnerability-scores",
        [
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
        "circl_vuln",
    ),
    (
        "rezaduty/cybersecurity-qa-v2",
        [
            "data/data-0000.parquet",
            "data/data-0001.parquet",
            "data/data-0002.parquet",
            "data/data-0003.parquet",
            "data/data-0004.parquet",
            "data/data-0005.parquet",
            "cybersecurity_interview_qa_complete.jsonl",
        ],
        "nvd_qa",
    ),
]


def download_files() -> None:
    for repo_id, paths, subdir in TARGETS:
        dest = RAW / subdir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {repo_id} -> {dest} ===")
        for path in paths:
            local = dest / Path(path).name
            if local.exists() and local.stat().st_size > 0:
                print(f"  [skip] {local.name} (already present)")
                continue
            print(f"  [get ] {path}")
            got = hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
                local_dir=dest,
            )
            print(f"  [ok  ] {got}")


def download_snapshot() -> None:
    """The PreTrain-640k dataset ships as ~100 small parquet shards."""
    repo_id = "11-47/Organized_PreTrain_Cyber_Security_640k"
    dest = RAW / "pretrain_640k"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".complete"
    if marker.exists():
        print(f"\n=== {repo_id} -> already downloaded ===")
        return
    print(f"\n=== {repo_id} -> {dest} ===")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=["cybersecurity/*.parquet", "README.md"],
        max_workers=8,
    )
    marker.write_text("ok\n")
    print("  [ok  ] snapshot complete")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        download_files()
        download_snapshot()
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run to resume (finished files are skipped)")
        return 130
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
