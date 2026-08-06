"""One-shot migration for AssemblyAI speaker labels.

Older runs of the AssemblyAI backend saved speaker IDs as ``SPEAKER_A`` /
``SPEAKER_B`` / …, while every other backend (and the current AssemblyAI
transcriber) uses zero-padded ``SPEAKER_00`` / ``SPEAKER_01`` / ….  This
script rewrites any non-canonical ``SPEAKER_*`` labels in cached translation
pickles and their sibling ``artifacts/dubbing_texts.tsv`` files so
`tts_to_end`, per-speaker voice mapping, reference-audio library, etc. all
match again.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\migrate_assemblyai_speaker_labels.py            # walk ./cache
    .\\.venv\\Scripts\\python.exe scripts\\migrate_assemblyai_speaker_labels.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\migrate_assemblyai_speaker_labels.py --cache-root D:\\some\\cache
    .\\.venv\\Scripts\\python.exe scripts\\migrate_assemblyai_speaker_labels.py --projects-root prj

A ``SPEAKER_XX`` label is treated as canonical when ``XX`` is exactly two
digits.  Anything else (``SPEAKER_A``, ``SPEAKER_1``, ``SPEAKER_100``) gets
remapped in order-of-first-appearance for each individual pickle.  The
mapping preserves whatever canonical labels already exist in the file, so
running the script twice is a no-op.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CANONICAL_RE = re.compile(r"^SPEAKER_\d{2}$")

DUBBING_TEXT_HEADER = [
    "speaker",
    "start",
    "end",
    "original",
    "translation",
    "synthesized_text",
    "audio_file",
]


def is_canonical(label: str) -> bool:
    return bool(CANONICAL_RE.match(label or ""))


def _next_free_slot(used: set[int]) -> int:
    slot = 0
    while slot in used:
        slot += 1
    return slot


def build_speaker_mapping(segments: Iterable[dict]) -> Dict[str, str]:
    """Return raw-label → canonical-label mapping for one pickle.

    Canonical labels already present in the file keep their slot; every
    non-canonical label is assigned the next free ``SPEAKER_NN`` in
    order-of-first-appearance.
    """
    mapping: Dict[str, str] = {}
    used_slots: set[int] = set()

    # Pass 1: reserve slots already used by canonical labels so we don't collide.
    for segment in segments:
        raw = str(segment.get("speaker", "") or "")
        if is_canonical(raw):
            mapping.setdefault(raw, raw)
            used_slots.add(int(raw.split("_", 1)[1]))

    # Pass 2: assign fresh slots to non-canonical labels in first-seen order.
    for segment in segments:
        raw = str(segment.get("speaker", "") or "")
        if not raw or raw in mapping:
            continue
        slot = _next_free_slot(used_slots)
        used_slots.add(slot)
        mapping[raw] = f"SPEAKER_{slot:02d}"

    return mapping


def apply_mapping_to_segments(segments: List[dict], mapping: Dict[str, str]) -> int:
    """Rewrite speaker labels in-place. Returns count of changed segments."""
    changed = 0
    for segment in segments:
        raw = str(segment.get("speaker", "") or "")
        new = mapping.get(raw, raw)
        if new != raw:
            segment["speaker"] = new
            changed += 1
    return changed


def rewrite_dubbing_texts_tsv(path: Path, mapping: Dict[str, str]) -> int:
    """Rewrite the speaker column of a sibling dubbing_texts.tsv.

    Returns the number of data rows whose speaker changed. Missing files are
    ignored (returns 0); malformed files are skipped with a warning.
    """
    if not path.is_file():
        return 0

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        print(f"  ! Could not read {path}: {exc}", file=sys.stderr)
        return 0

    if not rows:
        return 0

    header = rows[0]
    data_rows = rows[1:]
    if header != DUBBING_TEXT_HEADER:
        print(f"  ! Unexpected TSV header in {path}; leaving file untouched", file=sys.stderr)
        return 0

    changed = 0
    for row in data_rows:
        if not row:
            continue
        raw = row[0]
        new = mapping.get(raw, raw)
        if new != raw:
            row[0] = new
            changed += 1

    if changed == 0:
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(data_rows)
    return changed


def find_sibling_tsvs(pickle_path: Path, projects_root: Path | None) -> List[Path]:
    """Best-effort locate every dubbing_texts.tsv that mirrors this pickle.

    Pickles live at ``cache/<input-hash>/translation/<key>.pkl`` and have no
    direct backlink to the project. When ``--projects-root`` is provided we
    also touch every ``<projects_root>/*/artifacts/dubbing_texts.tsv`` — the
    speaker set is the same for every project that shares this pickle, so
    rewriting them all with the same mapping is safe.
    """
    tsvs: List[Path] = []
    if projects_root and projects_root.is_dir():
        for project_dir in sorted(projects_root.iterdir()):
            candidate = project_dir / "artifacts" / "dubbing_texts.tsv"
            if candidate.is_file():
                tsvs.append(candidate)
    return tsvs


def process_pickle(
    pickle_path: Path,
    projects_root: Path | None,
    dry_run: bool,
) -> Tuple[int, int]:
    """Rewrite one translation pickle plus every mirrored TSV.

    Returns ``(pickle_changes, tsv_changes)``.
    """
    try:
        with pickle_path.open("rb") as handle:
            segments = pickle.load(handle)
    except (OSError, pickle.UnpicklingError) as exc:
        print(f"  ! Skipping {pickle_path}: {exc}", file=sys.stderr)
        return 0, 0

    if not isinstance(segments, list):
        print(f"  ! Skipping {pickle_path}: unexpected payload type {type(segments).__name__}", file=sys.stderr)
        return 0, 0

    mapping = build_speaker_mapping(segments)
    non_canonical = {k: v for k, v in mapping.items() if k != v}
    if not non_canonical:
        return 0, 0

    print(f"  Remapping in {pickle_path}:")
    for raw, canonical in sorted(non_canonical.items()):
        print(f"    {raw} -> {canonical}")

    pickle_changes = apply_mapping_to_segments(segments, mapping)

    tsv_changes = 0
    if not dry_run:
        backup = pickle_path.with_suffix(pickle_path.suffix + ".bak")
        shutil.copy2(pickle_path, backup)
        with pickle_path.open("wb") as handle:
            pickle.dump(segments, handle)

        for tsv_path in find_sibling_tsvs(pickle_path, projects_root):
            tsv_changes += rewrite_dubbing_texts_tsv(tsv_path, non_canonical)

    return pickle_changes, tsv_changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default="cache",
        help="Root of the DubbLM cache/ tree (default: cache)",
    )
    parser.add_argument(
        "--projects-root",
        default="prj",
        help="Root of per-video project directories, used to find dubbing_texts.tsv (default: prj)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without touching any files",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    projects_root = Path(args.projects_root) if args.projects_root else None

    if not cache_root.is_dir():
        print(f"Cache root {cache_root} does not exist", file=sys.stderr)
        return 2

    pickles = sorted(cache_root.glob("*/translation/*.pkl"))
    if not pickles:
        print(f"No translation pickles under {cache_root}")
        return 0

    print(f"Scanning {len(pickles)} translation pickle(s) under {cache_root}...")
    total_pickle_changes = 0
    total_tsv_changes = 0
    touched_pickles = 0

    for pickle_path in pickles:
        pickle_changes, tsv_changes = process_pickle(pickle_path, projects_root, args.dry_run)
        if pickle_changes:
            touched_pickles += 1
            total_pickle_changes += pickle_changes
            total_tsv_changes += tsv_changes

    prefix = "[dry-run] would rewrite" if args.dry_run else "Rewrote"
    print(
        f"\n{prefix} {total_pickle_changes} segment(s) across {touched_pickles} pickle(s); "
        f"{total_tsv_changes} row(s) in dubbing_texts.tsv."
    )
    if not args.dry_run and touched_pickles:
        print("Backups saved next to each modified file with a .bak suffix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
