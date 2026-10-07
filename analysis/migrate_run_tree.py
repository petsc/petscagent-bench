"""Bring a results directory up to the layout the green agent writes today.

A recorded set made before commit 4253579 keeps its code under a per-run tree,
``runs/<purple>-judged-by-<judge>-run<N>/``, or under ``sources/`` before the
rename that preceded it. Today the two halves live apart: the code under
``code/<variant>/``, shared by every judge that scores the variant, and each
problem's score under ``scores/<variant>/<problem>/judged-by-<judge>.json``.
The aggregate changed with them, trading ``run_index`` for the ``pass_index``
that numbers a rescore, and gaining ``submissions`` and ``scored_at``.

Nothing on main reads the old names any more, so a set left on the old layout
is invisible to anything that walks a results directory. This rewrites one in
place to match what a fresh run produces.

Timestamps are the one thing that cannot be recovered: a run recorded before
``scored_at`` existed never wrote the time it was scored, so the field is left
null rather than guessed from a file's mtime. ``scored_by`` is recovered, being
the aggregate's own ``judge_model``.

Usage:
    python analysis/migrate_run_tree.py output_junchao [--dry-run]
"""

import argparse
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.green_agent.agent import BenchmarkResult, _name_slug, _slug

# The order asdict() gives a result record, so a migrated record reads the same
# as a freshly written one instead of merely holding the same keys.
RECORD_FIELDS = [f.name for f in fields(BenchmarkResult)]

# Where the per-run code tree lived, newest name first.
LEGACY_TREES = ("runs", "sources")


def _legacy_tree(root, stem):
    """The per-run code tree an aggregate was written with, or None."""
    found = [root / name / stem for name in LEGACY_TREES]
    return next((path for path in found if path.is_dir()), None)


def _spent_scores(tree, data):
    """The result.json files in a tree that the aggregate already accounts for.

    A tree written under the ``runs/`` layout keeps each problem's score beside
    its source as ``result.json``. The code tree holds only sources and a
    manifest, so moving one wholesale would carry a file there that nothing
    lists and no later run overwrites. They are redundant, being the same
    records the aggregate holds, so this returns the ones that match to be
    dropped after the move. One that disagrees is returned separately and kept,
    because a disagreement is a difference worth looking at rather than
    deleting.
    """
    records = {r["problem_name"]: r for r in data["results"]}
    spent, conflicting = [], []
    for path in sorted(tree.glob("*/result.json")):
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            conflicting.append(path)
            continue
        recorded = records.get(found.get("problem_name"))
        (spent if recorded == found else conflicting).append(path)
    return spent, conflicting


def _rename(mapping, target):
    """Move each old key to its new name, in place, where the old one is used.

    A key already under its new name is left alone, so this is safe to run
    over a record that has been migrated or was never on the old name.
    """
    if not isinstance(target, dict):
        return
    for old, new in mapping.items():
        if old in target and new not in target:
            target[new] = target.pop(old)


# Keys the green agent renamed rather than retired. Commit c7c2460 replaced
# the purple response cache with run replay, which took over what the cache
# had counted: a response that was not generated fresh for this pass. The
# record flag and the count beside it were renamed with it, so a record older
# than the rename holds today's value under yesterday's name.
RECORD_RENAMES = {"purple_response_from_cache": "purple_response_replayed"}
MEASURED_RENAMES = {"cached_cases": "replayed_cases"}


def _migrate_summary(summary):
    """The run summary in today's shape."""
    if isinstance(summary, dict):
        _rename(MEASURED_RENAMES,
                summary.get("purple_efficiency", {}).get("benchmark_measured"))
    return summary


def _migrate_record(record, judge_model):
    """One result record in today's field order.

    Returns the record and the names of any fields it has no value for, which
    the caller reports: a record short of a field is a record this script was
    not written for, and filling it with a default would invent data.
    """
    migrated = {"scored_at": None, "scored_by": judge_model, **record}
    _rename(RECORD_RENAMES, migrated)
    missing = [name for name in RECORD_FIELDS if name not in migrated]
    ordered = {name: migrated[name] for name in RECORD_FIELDS if name in migrated}
    # Anything the dataclass has since dropped is kept rather than discarded,
    # so migrating never loses a field this script does not know about.
    ordered.update({k: v for k, v in migrated.items() if k not in ordered})
    return ordered, missing


def _migrate_aggregate(data, variant, judge_model):
    """The aggregate in today's shape, and the per-problem records it holds."""
    records = []
    missing = set()
    for record in data["results"]:
        migrated, absent = _migrate_record(record, judge_model)
        records.append(migrated)
        missing.update(absent)
    return {
        "agent": data.get("agent"),
        "purple_model": data.get("purple_model"),
        "reported_model": data.get("reported_model"),
        "judge_model": judge_model,
        # Null, because a recorded run is a live pass rather than a rescore.
        # run_index, which it replaces, is dropped here.
        "pass_index": None,
        "submissions": f"code/{variant}",
        "problem_filter": data.get("problem_filter"),
        # Not recoverable; see the module docstring.
        "scored_at": None,
        "summary": _migrate_summary(data.get("summary")),
        "results": records,
    }, sorted(missing)


def _migrate_one(root, aggregate, dry_run):
    """Migrate one run. Returns True if anything about it changed."""
    try:
        data = json.loads(aggregate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"  WARN  {aggregate.name}  (not valid JSON, left alone)")
        return False
    if not isinstance(data, dict) or "results" not in data:
        return False
    if "pass_index" in data:
        print(f"  skip  {aggregate.name}  (already migrated)")
        return False

    judge_model = data.get("judge_model")
    variant = _name_slug(data.get("reported_model") or data.get("purple_model"))
    judge_slug = _name_slug(judge_model)
    code_dir = root / "code" / variant
    scores_dir = root / "scores" / variant
    target = root / f"{variant}-judged-by-{judge_slug}.json"

    tree = _legacy_tree(root, aggregate.stem)
    if code_dir.exists():
        print(f"  SKIP  {aggregate.name}  (code/{variant} already exists)")
        return False
    if target != aggregate and target.exists():
        print(f"  SKIP  {aggregate.name}  (would overwrite {target.name})")
        return False

    migrated, missing = _migrate_aggregate(data, variant, judge_model)
    for name in missing:
        print(f"  WARN  {aggregate.name}  (records have no {name})")

    spent, conflicting = ([], []) if tree is None else _spent_scores(tree, data)
    for path in conflicting:
        print(f"  WARN  {aggregate.name}  ({path.name} in "
              f"{path.parent.name} disagrees with the aggregate, kept)")

    print(f"  move  {aggregate.name}")
    if tree is None:
        print("          (no code tree found, scores only)")
    else:
        print(f"          {tree.relative_to(root)} -> code/{variant}")
    if spent:
        print(f"          drop {len(spent)} superseded result.json")
    print(f"          {len(migrated['results'])} scores -> "
          f"scores/{variant}/*/judged-by-{judge_slug}.json")
    if target != aggregate:
        print(f"          {aggregate.name} -> {target.name}")
    if dry_run:
        return True

    if tree is not None:
        code_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tree), str(code_dir))
        # After the move, so the tree is dropped in one step and a failure
        # leaves the old layout whole rather than half stripped.
        for path in spent:
            (code_dir / path.relative_to(tree)).unlink()
    # Every result gets a score file, including a problem the agent produced
    # no code for, which is what the green agent does: it writes the scores
    # from the results and the sources from generated_sources, separately.
    for record in migrated["results"]:
        problem_dir = scores_dir / _slug(record["problem_name"])
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / f"judged-by-{judge_slug}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    target.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
    if target != aggregate:
        aggregate.unlink()
    return True


def migrate(root, dry_run):
    """Migrate one results directory. Returns the number of runs changed."""
    changed = sum(
        _migrate_one(root, aggregate, dry_run)
        for aggregate in sorted(root.glob("*.json"))
    )
    for name in LEGACY_TREES:
        legacy = root / name
        if legacy.is_dir() and not any(legacy.iterdir()):
            print(f"  remove empty {name}/")
            if not dry_run:
                legacy.rmdir()
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="results directories to migrate")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without changing it")
    args = ap.parse_args(argv)

    total = 0
    for name in args.dirs:
        root = Path(name)
        if not root.is_dir():
            print(f"{name}: not a directory")
            continue
        print(f"{name}:")
        total += migrate(root, args.dry_run)
        print()
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"{verb} {total} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
