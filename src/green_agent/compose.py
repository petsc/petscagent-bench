"""Build an aggregate from a score tree, for the runs that wrote no aggregate."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.green_agent.agent import (
    _name_slug,
    _reported_model,
    _result_from_record,
    _summarize,
)

_SCORE_NAME = "judged-by-"


def _problem_order(record):
    pid = record.get("problem_id")
    try:
        return (0, int(pid), "")
    except (TypeError, ValueError):
        return (1, 0, str(pid))


def _identity_from_siblings(output_dir: Path, variant: str, judge: str):
    """`agent` and `purple_model` describe the launch, so no record holds them.

    Taken from an aggregate already written for this variant and judge,
    preferring one that names a model. Read before anything is written, since
    the canonical name is one of the candidates.
    """
    aggregates = []
    for path in sorted(output_dir.glob(f"{variant}-judged-by-{judge}*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "results" in data:
            aggregates.append(data)
    for data in aggregates:
        if data.get("purple_model"):
            return data.get("agent"), data.get("purple_model")
    for data in aggregates:
        return data.get("agent"), data.get("purple_model")
    return None, None


def compose_one(
    output_dir: Path, variant: str, judge: str, score_files: List[Path],
    efficiency_config: Optional[Dict[str, Any]] = None,
    identity: Optional[tuple] = None,
) -> Path:
    """Write one variant's aggregate for one judge. Returns the path."""
    # Written back out as they came in, so a field this checkout does not know
    # survives. Only the objects feeding _summarize are filtered to the
    # dataclass.
    records = [json.loads(p.read_text(encoding="utf-8")) for p in score_files]
    records.sort(key=_problem_order)
    results = [_result_from_record(r) for r in records]

    agent_id, purple_model = identity or _identity_from_siblings(
        output_dir, variant, judge
    )
    json_data = {
        "agent": agent_id,
        "purple_model": purple_model,
        "reported_model": _reported_model(results),
        # The filename holds a slug; scored_by holds the model string.
        "judge_model": next(
            (r.get("scored_by") for r in records if r.get("scored_by")), judge
        ),
        "pass_index": None,
        "submissions": f"code/{variant}",
        "problem_filter": None,
        # The newest pass represented, so this still says when the scores
        # happened rather than when they were gathered up.
        "scored_at": max((r.get("scored_at") or "" for r in records),
                         default="") or None,
        "composed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": _summarize(results, efficiency_config),
        "results": records,
    }
    path = output_dir / f"{variant}-judged-by-{judge}.json"
    path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    return path


def compose_variant(
    output_dir, variant: str, judge: str,
    efficiency_config: Optional[Dict[str, Any]] = None,
    identity: Optional[tuple] = None,
) -> Optional[Path]:
    """Compose one variant and judge, leaving every other aggregate alone.

    `identity` supplies `agent` and `purple_model` for a caller that knows
    them, since the tree does not hold them.
    """
    output_dir = Path(output_dir)
    judge_slug = _name_slug(judge)
    score_files = sorted(
        (output_dir / "scores" / variant).glob(f"*/{_SCORE_NAME}{judge_slug}.json")
    )
    if not score_files:
        return None
    return compose_one(output_dir, variant, judge_slug, score_files,
                       efficiency_config, identity)


def compose(output_dir, judge: Optional[str] = None,
            efficiency_config: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Compose an aggregate per variant and judge found under output_dir.

    `judge` narrows to one, named as a slug or as the full model string.
    Returns the paths written.
    """
    output_dir = Path(output_dir)
    scores_root = output_dir / "scores"
    if not scores_root.is_dir():
        raise FileNotFoundError(f"no score tree at {scores_root}")

    written = []
    for variant_dir in sorted(p for p in scores_root.iterdir() if p.is_dir()):
        # One tree can hold several judges' verdicts on the same problems.
        by_judge: Dict[str, List[Path]] = {}
        for score_file in sorted(variant_dir.glob(f"*/{_SCORE_NAME}*.json")):
            by_judge.setdefault(score_file.stem[len(_SCORE_NAME):],
                                []).append(score_file)
        for found, score_files in sorted(by_judge.items()):
            if judge and found != _name_slug(judge):
                continue
            written.append(compose_one(output_dir, variant_dir.name, found,
                                       score_files, efficiency_config))
    return written
