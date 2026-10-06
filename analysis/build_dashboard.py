#!/usr/bin/env python3
"""Build the self-contained HTML results dashboard.

Reads every provenanced result file and renders two files:

    analysis/dashboard.html           standalone page, open it locally
    analysis/dashboard_artifact.html  same page without the document skeleton,
                                      for publishing as a Claude artifact

Both come from analysis/dashboard_template.html, so the local copy and the
shared copy can never drift. Usage:

    python analysis/build_dashboard.py
    python analysis/build_dashboard.py --demo      # synthetic six-variant set
    python analysis/build_dashboard.py --dir path  # any directory of results

The purple axis is a factor model, so the payload carries the factors a variant
is made of and the contrasts running between them, not just display names. The
dashboard colors by one factor and facets by the rest, which is what keeps the
categorical palette inside its validated budget as variants are added.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from load_results import (
    CATEGORY_WEIGHTS,
    EVALUATOR_CATEGORY,
    FACTOR_LABELS,
    FACTORS,
    FAILURE_CLASSES,
    REPO_ROOT,
    TIER_ORDER,
    Record,
    bootstrap_ci,
    correlation,
    find_contrasts,
    label_for,
    load_records,
    paired_bootstrap,
    variance_components,
)

# Harness exception text runs to ~19k characters for a PETSc stack dump. The
# drill-down shows the head of it, which carries the signal, and names the full
# length so nothing looks silently complete.
ERROR_CHARS = 4000

# Generated sources are shown in full up to this size; past it the file is cut
# and the cut is stated. A PETSc solution is normally 3-12k.
SOURCE_CHARS = 60_000

# Per-case stderr, carried only for the cases that failed. A case that ran is
# diagnosed by nothing, and the passing cases' PETSc chatter would be most of
# the payload.
CASE_STDERR_CHARS = 1200

# A contrast needs this many shared problems before the page will draw it. The
# bootstrap resamples problems, so at one shared problem every draw is that same
# problem and the interval collapses to zero width. Such a row renders as a
# resolved result with perfect certainty, which is an artifact of the method
# rather than a finding. Three matches the floor `correlation` already applies,
# so a drawn row always carries an r as well.
MIN_PAIRED_PROBLEMS = 3

ANALYSIS_DIR = Path(__file__).resolve().parent
TEMPLATE = ANALYSIS_DIR / "dashboard_template.html"
STANDALONE_OUT = ANALYSIS_DIR / "dashboard.html"
ARTIFACT_OUT = ANALYSIS_DIR / "dashboard_artifact.html"

DEMO_DIR = Path("/tmp/petscbench_demo")

# Preferred order for base models, so a model keeps its position on the axis no
# matter which variants happen to be in view.
MODEL_ORDER = ("anthropic/claudeopus46", "openai/gpt52", "openai/gemini25pro")


def score(r: Record) -> float:
    return r.composite_score


def ran(r: Record) -> float:
    return 1.0 if r.ran else 0.0


def variant_sort_key(v) -> tuple:
    """Single agents first, then by scaffold size, then by model and skills.

    Ordering by configuration rather than by score keeps each variant in the
    same column when the judge filter changes; color and position follow the
    entity, never its current rank.
    """
    try:
        model_rank = MODEL_ORDER.index(v.base_model)
    except ValueError:
        model_rank = len(MODEL_ORDER)
    return (0 if v.scaffold == "single" else 1, v.n_subagents, model_rank,
            len(v.skills), v.variant_id)


def build_payload(dirs: list[Path] | None) -> dict:
    records = load_records(dirs, max_source_bytes=SOURCE_CHARS)
    if not records:
        raise SystemExit("No provenanced result files found.")

    variants = sorted({r.variant for r in records}, key=variant_sort_key)
    judges = sorted({r.judge for r in records})
    problems = sorted({r.problem for r in records})
    variant_ix = {v.variant_id: i for i, v in enumerate(variants)}
    judge_ix = {j: i for i, j in enumerate(judges)}
    problem_ix = {p: i for i, p in enumerate(problems)}

    rows = []
    for rec in records:
        rows.append(
            {
                "v": variant_ix[rec.variant.variant_id],
                "j": judge_ix[rec.judge],
                "r": rec.run_index,
                "p": problem_ix[rec.problem],
                "s": round(rec.composite_score, 2),
                "t": rec.tier,
                "comp": rec.compiles,
                "ran": rec.ran,
                "gp": rec.gates_passed,
                "gt": rec.gates_total,
                "cat": {k: round(float(v), 2) for k, v in rec.categories.items()},
                "cli": rec.cli_args,
                "wt": round(rec.wall_time_sec, 2),
                "tok": rec.total_tokens,
                "src": rec.source_file,
                "fc": rec.failure_class,
                "gf": rec.gate_failed,
                "err": rec.error[:ERROR_CHARS],
                "errlen": len(rec.error),
                "cases": [
                    {
                        "i": c.index,
                        "a": c.args,
                        "n": c.nsize,
                        "ok": c.runs,
                        "t": round(c.execution_time_sec, 3)
                        if c.execution_time_sec is not None else None,
                        "e": "" if c.runs else c.stderr[:CASE_STDERR_CHARS],
                        "elen": 0 if c.runs else len(c.stderr),
                    }
                    for c in rec.cases
                ],
                "code": [
                    {"f": s.filename, "h": s.sha256[:12], "t": s.text}
                    for s in rec.sources
                ],
                "ev": [
                    {
                        "n": e.name,
                        "t": e.type,
                        "m": e.method,
                        "p": e.passed,
                        "s": e.score,
                        "c": e.confidence,
                        "f": e.feedback,
                    }
                    for e in rec.evaluations
                ],
            }
        )

    # Only factors that actually vary are offered as a color axis. A factor with
    # one level would spend a categorical slot to say nothing.
    factors = []
    for f in FACTORS:
        levels = []
        for v in variants:
            key = str(v.factor(f))
            if key not in [lv["value"] for lv in levels]:
                levels.append({"value": key, "label": v.factor_label(f)})
        if len(levels) > 1:
            factors.append({"key": f, "label": FACTOR_LABELS[f], "levels": levels})
    varying = [f["key"] for f in factors]

    slices: dict[str, list[Record]] = {"all": records}
    for j in judges:
        slices[j] = [r for r in records if r.judge == j]

    # Bootstrap intervals are precomputed per judge slice so the page never has
    # to resample in the browser, and so the figures match the paper exactly.
    ci: dict[str, dict[str, dict[str, list[float]]]] = {}
    for key, subset in slices.items():
        ci[key] = {}
        for v in variants:
            mine = [r for r in subset if r.variant == v]
            s_mean, s_lo, s_hi = bootstrap_ci(mine, score)
            r_mean, r_lo, r_hi = bootstrap_ci(mine, ran)
            ci[key][v.variant_id] = {
                "score": [round(s_mean, 2), round(s_lo, 2), round(s_hi, 2)],
                "rate": [round(r_mean, 4), round(r_lo, 4), round(r_hi, 4)],
            }

    # Contrasts are paired on problem, which is the whole reason a six-problem
    # suite can resolve a one-factor change. The correlation ships with each one
    # because pairing only buys precision when the arms move together.
    contrast_defs = find_contrasts(variants)
    contrasts: dict[str, list[dict]] = {}
    for key, subset in slices.items():
        out = []
        for c in contrast_defs:
            test = [r for r in subset if r.variant == c.test]
            base = [r for r in subset if r.variant == c.base]
            d, lo, hi, n = paired_bootstrap(test, base, score)
            rd, rlo, rhi, _ = paired_bootstrap(test, base, ran)
            if n == 0:
                continue
            out.append(
                {
                    "withheld": n < MIN_PAIRED_PROBLEMS,
                    "key": c.key,
                    "factor": c.factor,
                    "factorLabel": FACTOR_LABELS[c.factor],
                    "label": c.label,
                    "arms": c.arms,
                    "held": c.held,
                    "base": variant_ix[c.base.variant_id],
                    "test": variant_ix[c.test.variant_id],
                    "d": [round(d, 2), round(lo, 2), round(hi, 2)],
                    "rate": [round(rd, 4), round(rlo, 4), round(rhi, 4)],
                    "n": n,
                    "r": round(correlation(test, base, score), 3),
                }
            )
        contrasts[key] = out

    # Where the next unit of compute should go. Between-problem variance that
    # dominates means extra runs of the same problems buy almost nothing.
    variance = {}
    for key, subset in slices.items():
        between, within = variance_components(subset, score)
        n_problems = len({r.problem for r in subset})
        n_runs = len({r.run_index for r in subset})
        variance[key] = {
            "between": round(between, 1),
            "within": round(within, 1),
            "problems": n_problems,
            "runs": n_runs,
        }

    harness_revs = sorted({r.harness_rev for r in records if r.harness_rev})
    petsc_revs = sorted({r.petsc_rev for r in records if r.petsc_rev})
    n_code = sum(1 for r in records if r.sources)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "variants": [
            {
                "id": v.variant_id,
                "label": v.label,
                "short": v.short_label(varying),
                "factors": {f: str(v.factor(f)) for f in FACTORS},
                "flabels": {f: v.factor_label(f) for f in FACTORS},
            }
            for v in variants
        ],
        "factors": factors,
        "factorLabels": FACTOR_LABELS,
        "judges": [{"id": j, "label": label_for(j)} for j in judges],
        "problems": problems,
        "categories": list(CATEGORY_WEIGHTS),
        "weights": CATEGORY_WEIGHTS,
        "evalCategory": EVALUATOR_CATEGORY,
        "tiers": list(TIER_ORDER),
        "failure_classes": [c for c in FAILURE_CLASSES
                            if any(r["fc"] == c for r in rows)],
        "rows": rows,
        "ci": ci,
        "contrasts": contrasts,
        "variance": variance,
        "harnessRevs": harness_revs,
        "petscRevs": petsc_revs,
        "codeRuns": n_code,
        "minPaired": MIN_PAIRED_PROBLEMS,
        "nProblems": len(problems),
    }


def _nulls_for_nan(obj):
    """Replace non-finite floats with None throughout the payload.

    The statistics deliberately return nan for "not computable": a correlation
    needs three shared problems, a bootstrap needs at least one. A variant with
    a single run, or none under one judge, hits both. JSON has no nan, so the
    page reads those as null and the formatters already render null as an em
    dash.
    """
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _nulls_for_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nulls_for_nan(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="build against the synthetic set in /tmp/petscbench_demo")
    ap.add_argument("--dir", type=Path, action="append",
                    help="result directory to read (repeatable)")
    args = ap.parse_args()

    dirs = None
    if args.demo:
        dirs = [DEMO_DIR]
    elif args.dir:
        dirs = args.dir

    payload = _nulls_for_nan(build_payload(dirs))
    # `</script>` inside evaluator feedback or generated code would close the
    # data block early. `allow_nan=False` keeps a non-finite float from reaching
    # the page as a bare `NaN` token, which is not valid JSON and would fail the
    # whole `JSON.parse` rather than degrade one card.
    blob = json.dumps(payload, separators=(",", ":"),
                      allow_nan=False).replace("</", "<\\/")

    template = TEMPLATE.read_text()
    if "__PAYLOAD__" not in template:
        raise SystemExit(f"{TEMPLATE} is missing the __PAYLOAD__ placeholder.")
    body = template.replace("__PAYLOAD__", blob)

    ARTIFACT_OUT.write_text(body)
    STANDALONE_OUT.write_text(STANDALONE_HEAD + "</head>\n<body>\n" + body + STANDALONE_TAIL)

    n_runs = len(payload["rows"])
    n_var = len(payload["variants"])
    n_con = len(payload["contrasts"]["all"])
    for path in (STANDALONE_OUT, ARTIFACT_OUT):
        size = path.stat().st_size / 1024
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"{shown}  {size:,.0f} KB")
    print(f"{n_runs} problem-runs · {n_var} variants · {n_con} contrasts · "
          f"{payload['codeRuns']} runs with source")


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  :root { color-scheme: light; }
  body { margin: 0; font: 14px system-ui, -apple-system, "Segoe UI", sans-serif; background: #f9f9f7; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
"""

STANDALONE_TAIL = """
</body>
</html>
"""


if __name__ == "__main__":
    main()
