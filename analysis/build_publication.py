#!/usr/bin/env python3
"""Build rerunnable, publication-ready figures and tables from benchmark results.

The result directory is rescanned on every invocation.  Outputs are assembled
in a temporary sibling directory and swapped into place only after every file
has been generated, so an interrupted build leaves the previous bundle valid.

Examples:
    python analysis/build_publication.py
    python analysis/build_publication.py --dir output --out publication
    python analysis/build_publication.py --dir output --pooled
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from load_results import (
    CATEGORY_WEIGHTS,
    REPO_ROOT,
    SOURCE_DIRS,
    TIER_ORDER,
    Record,
    Variant,
    label_for,
    load_records,
)

MIN_FIGURE_PROBLEMS = 3
DEFAULT_OUT = REPO_ROOT / "publication"
NAVY = "#24445F"
PALE_BLUE = "#E8F1F7"
PALE_TEAL = "#DDEFEA"
PALE_CORAL = "#F7E1DC"
PALE_GOLD = "#F5ECD7"

# Headless benchmark hosts often mount the home directory read-only.  Keep
# Matplotlib and fontconfig caches out of it so a normal build is quiet and
# deterministic in CI as well as on a workstation.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "petscagent-publication-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))


def _mpl():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Publication figures require matplotlib. Install it with "
            "`uv add matplotlib` or run in an environment that provides it."
        ) from exc
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    return plt


def _per_problem(records: Iterable[Record], value: Callable[[Record], float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.problem].append(value(record))
    return {name: sum(values) / len(values) for name, values in grouped.items()}


def _variant_order(records: Iterable[Record]) -> list[Variant]:
    return sorted({r.variant for r in records}, key=lambda v: v.variant_id)


def _short_labels(variants: list[Variant]) -> list[str]:
    varying = [
        factor for factor in ("scaffold", "base_model", "skills", "n_subagents")
        if len({v.factor(factor) for v in variants}) > 1
    ]
    labels = [v.short_label(varying) for v in variants]
    # Legacy flat tags can share a long experiment prefix and differ only in a
    # short arm suffix.  Repeating that prefix on every tick wastes most of a
    # single-column figure and can make adjacent labels collide.  Compact only
    # when the remaining suffixes are non-empty and still unique; full names
    # remain in both publication tables and source-data files.
    if len(labels) > 1 and max(map(len, labels)) > 18:
        prefix = os.path.commonprefix(labels)
        boundary = max(prefix.rfind("-"), prefix.rfind("/"), prefix.rfind("_"))
        if boundary >= 0:
            compact = [label[boundary + 1:] for label in labels]
            if all(compact) and len(set(compact)) == len(compact):
                labels = compact
    return labels


def _save_figure(fig, stem: Path) -> None:
    """Export the publication figure as an editable PDF."""
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _overview(records: list[Record]) -> list[dict]:
    variants = _variant_order(records)
    labels = _short_labels(variants)
    rows = []
    for variant, label in zip(variants, labels):
        mine = [r for r in records if r.variant == variant]
        # The problem set is curated, not sampled, so this is a measurement
        # and not an estimate carrying an interval.
        mean = sum(r.composite_score for r in mine) / len(mine)
        run_mean = sum(float(r.ran) for r in mine) / len(mine)

        n_judge_passes = max((len(r.pass_scores) for r in mine), default=1)

        def mean_present(values):
            present = [float(value) for value in values if value is not None]
            return sum(present) / len(present) if present else None

        def sample_sd(values):
            present = [float(value) for value in values if value is not None]
            if len(present) < 2:
                return None
            center = sum(present) / len(present)
            return math.sqrt(sum((value - center) ** 2 for value in present)
                             / (len(present) - 1))

        def across_passes(value_of):
            # Pooling problems and passes would report between-problem spread
            # whenever there is only one pass.
            means = []
            for pass_i in range(n_judge_passes):
                means.append(mean_present([value_of(r, pass_i) for r in mine]))
            return sample_sd(means)

        judge_sd = across_passes(
            lambda r, i: r.pass_scores[i] if i < len(r.pass_scores) else None
        )

        rows.append({
            "variant_id": variant.variant_id,
            "variant": variant.label,
            "plot_label": label,
            "mean_score": mean,
            "execution_rate": run_mean,
            "gold_rate": sum(r.tier == "GOLD" for r in mine) / len(mine),
            "n_problems": len({r.problem for r in mine}),
            "n_problem_runs": len(mine),
            "n_judge_passes": n_judge_passes,
            "judge_score_sd": judge_sd,
            "problems": sorted({r.problem for r in mine}),
            **{
                f"category_{category}": mean_present(
                    [r.categories.get(category) for r in mine]
                )
                for category in CATEGORY_WEIGHTS
            },
            **{
                f"category_{category}_sd": across_passes(
                    lambda r, i, c=category: r.pass_category_scores[i].get(c)
                    if i < len(r.pass_category_scores) else None
                )
                for category in CATEGORY_WEIGHTS
            },
            "cost_usd_per_run": mean_present([r.agent_cost_usd for r in mine]),
            "prompt_tokens_per_run": mean_present([r.prompt_tokens for r in mine]),
            "completion_tokens_per_run": mean_present([r.completion_tokens for r in mine]),
            "cached_tokens_per_run": mean_present([r.cached_tokens for r in mine]),
            "total_tokens_per_run": mean_present([r.total_tokens for r in mine]),
            "model_calls_per_run": mean_present([r.model_calls for r in mine]),
            "tool_calls_per_run": mean_present([r.tool_calls for r in mine]),
            "peak_context_tokens": mean_present([r.peak_context_tokens for r in mine]),
            "execution_time_sec": mean_present([r.execution_time_sec for r in mine]),
            "agent_wall_time_sec": mean_present([r.agent_wall_time_sec for r in mine]),
            "all_replayed": all(r.response_replayed for r in mine),
        })

    return rows


def _matrix(records: list[Record], out: Path, render: bool = True) -> list[dict]:
    variants = _variant_order(records)
    problems = sorted({r.problem for r in records})
    labels = _short_labels(variants)
    values = {(v.variant_id, p): score for v in variants
              for p, score in _per_problem(
                  (r for r in records if r.variant == v),
                  lambda r: r.composite_score).items()}
    plot_labels = dict(zip((v.variant_id for v in variants), labels))
    rows = [
        {"problem": p, "variant_id": v.variant_id, "variant": v.label,
         "plot_label": plot_labels[v.variant_id],
         "mean_score": values.get((v.variant_id, p))}
        for p in problems for v in variants
    ]
    if not render:
        return rows
    plt = _mpl()
    matrix = [[values.get((v.variant_id, p), math.nan) for v in variants] for p in problems]
    fig_width = max(3.5, 0.48 * len(variants) + 1.7)
    fig_height = max(2.3, 0.32 * len(problems) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(variants)), labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(problems)), problems)
    for row_i, row in enumerate(matrix):
        for col_i, value in enumerate(row):
            text = "—" if math.isnan(value) else f"{value:.0f}"
            color = "white" if not math.isnan(value) and value >= 55 else "#222222"
            ax.text(col_i, row_i, text, ha="center", va="center", color=color, fontsize=5)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    bar.set_label("Composite score")
    _save_figure(fig, out / "figure_problem_matrix")
    plt.close(fig)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluator_rows(records: list[Record]) -> list[dict]:
    """Return one publication row per evaluator/configuration combination."""
    variants = _variant_order(records)
    evaluator_order: list[str] = []
    evaluator_types: dict[str, str] = {}
    for record in records:
        for evaluation in record.evaluations:
            if evaluation.name not in evaluator_types:
                evaluator_order.append(evaluation.name)
                evaluator_types[evaluation.name] = evaluation.type

    rows = []
    for evaluator in evaluator_order:
        for variant in variants:
            mine = [record for record in records if record.variant == variant]
            n_passes = max((len(record.pass_evaluator_values) for record in mine), default=0)
            pass_means = []
            pass_counts = []
            for pass_i in range(n_passes):
                values = [
                    record.pass_evaluator_values[pass_i][evaluator]
                    for record in mine
                    if pass_i < len(record.pass_evaluator_values)
                    and evaluator in record.pass_evaluator_values[pass_i]
                ]
                if values:
                    pass_means.append(sum(values) / len(values))
                    pass_counts.append(len(values))
            mean = sum(pass_means) / len(pass_means) if pass_means else None
            sd = None
            if len(pass_means) >= 2:
                sd = math.sqrt(sum((value - mean) ** 2 for value in pass_means)
                               / (len(pass_means) - 1))
            rows.append({
                "evaluator": evaluator,
                "evaluator_type": evaluator_types[evaluator],
                "variant_id": variant.variant_id,
                "variant": variant.label,
                "mean": mean,
                "sd_across_judge_passes": sd,
                "n_judge_passes": len(pass_means),
                "n_problem_runs_per_pass": min(pass_counts) if pass_counts else 0,
            })
    return rows


def _style_table(table, header_color=NAVY, stripe_color="#F4F7F9") -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.35)
        elif row % 2 == 0:
            cell.set_facecolor(stripe_color)


def _render_summary_table(path: Path, rows: list[dict]) -> None:
    plt = _mpl()
    show_judge_sd = all(row["n_judge_passes"] >= 2 for row in rows)

    def number(value, digits=1, scale=1.0):
        return "—" if value is None else f"{value / scale:.{digits}f}"

    def score_with_sd(row, category):
        mean = row.get(f"category_{category}")
        sd = row.get(f"category_{category}_sd")
        if mean is None:
            return "—"
        return (f"{mean:.1f} ± {sd:.1f}"
                if show_judge_sd and sd is not None else f"{mean:.1f}")

    effectiveness = []
    resources = []
    for row in rows:
        if show_judge_sd:
            score = f"{row['mean_score']:.1f} ± {row['judge_score_sd']:.1f}"
        else:
            score = f"{row['mean_score']:.1f}"
        effectiveness.append([
            row["variant"], score,
            score_with_sd(row, "correctness"),
            score_with_sd(row, "performance"),
            score_with_sd(row, "code_quality"),
            score_with_sd(row, "algorithm"),
            score_with_sd(row, "petsc"),
        ])
        cached_share = None
        if row.get("prompt_tokens_per_run"):
            cached_share = 100 * row["cached_tokens_per_run"] / row["prompt_tokens_per_run"]
        resources.append([
            row["variant"], number(row.get("cost_usd_per_run"), 2),
            number(row.get("total_tokens_per_run"), 2, 1_000_000),
            number(cached_share, 1),
            number(row.get("model_calls_per_run"), 0),
            number(row.get("tool_calls_per_run"), 0),
            number(row.get("peak_context_tokens"), 1, 1_000),
        ])

    fig_height = max(3.0, 0.58 * (len(rows) + 1) + 1.15)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, fig_height))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.87, bottom=0.14, hspace=0.62)
    title = (f"{rows[0]['problems'][0]} — one fixed submission, "
             f"{rows[0]['n_judge_passes']} judge passes per configuration"
             if rows[0]["n_problems"] == 1 and show_judge_sd
             else f"{rows[0]['problems'][0]} — one problem-run per configuration"
             if rows[0]["n_problems"] == 1
             else f"Benchmark summary — {rows[0]['n_problems']} problems per configuration")
    fig.suptitle(title, fontsize=8, fontweight="bold", x=0.025, ha="left")
    for ax in axes:
        ax.set_axis_off()

    pass_note = (f"mean across {rows[0]['n_judge_passes']} judge passes; "
                 if show_judge_sd else "")
    axes[0].set_title(f"Effectiveness ({pass_note}0–100; higher is better)", loc="left", fontsize=7,
                      fontweight="bold", pad=5)
    table = axes[0].table(
        cellText=effectiveness,
        colLabels=["Configuration", "Overall\nmean ± SD" if show_judge_sd else "Overall",
                   "Correctness", "Performance", "Code quality", "Algorithm", "PETSc"],
        cellLoc="right",
        colLoc="right",
        colWidths=[0.30, 0.14, 0.12, 0.12, 0.12, 0.10, 0.10],
        edges="closed",
        loc="center",
    )
    for row in range(len(effectiveness) + 1):
        table[(row, 0)].get_text().set_ha("left")
    _style_table(table)
    # The metric families remain distinguishable without turning the table
    # into a rainbow: category columns use related low-saturation hues.
    metric_header_colors = [NAVY, "#315F7D", "#397A8C", "#577F78",
                            "#81745D", "#876B70", "#6C6688"]
    for column, color in enumerate(metric_header_colors):
        table[(0, column)].set_facecolor(color)

    axes[1].set_title("Resource use per problem-run", loc="left", fontsize=7,
                      fontweight="bold", pad=5)
    resource_table = axes[1].table(
        cellText=resources,
        colLabels=["Configuration", "Cost\n(USD)", "Tokens\n(M)", "Cached\n(%)",
                   "Model\ncalls", "Tool\ncalls", "Peak ctx.\n(k)"],
        cellLoc="right",
        colLoc="right",
        colWidths=[0.32, 0.12, 0.12, 0.12, 0.11, 0.11, 0.10],
        edges="closed",
        loc="center",
    )
    for row in range(len(resources) + 1):
        resource_table[(row, 0)].get_text().set_ha("left")
    _style_table(resource_table, header_color="#7A6335", stripe_color="#FAF6EC")

    fig.text(
        0.025, 0.045,
        "Cost and token/call counts are recorded agent telemetry. Agent wall time is "
        "unavailable for replayed submissions.",
        fontsize=5.5, ha="left", va="bottom",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_evaluator_table(path: Path, rows: list[dict]) -> None:
    """Render all metric scores and gate verdicts without crowding the summary."""
    plt = _mpl()
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    evaluators = list(dict.fromkeys(row["evaluator"] for row in rows))
    lookup = {(row["evaluator"], row["variant"]): row for row in rows}

    body = []
    for evaluator in evaluators:
        sample = next(row for row in rows if row["evaluator"] == evaluator)
        is_gate = sample["evaluator_type"] == "gate"
        cells = [evaluator.replace("_", " "), "Gate" if is_gate else "Score"]
        for variant in variants:
            row = lookup.get((evaluator, variant))
            if not row or row["mean"] is None:
                cells.append("—")
            elif is_gate:
                cells.append(f"{100 * row['mean']:.0f}% pass")
            elif row["sd_across_judge_passes"] is None:
                cells.append(f"{row['mean']:.3f}")
            else:
                cells.append(
                    f"{row['mean']:.3f} ± {row['sd_across_judge_passes']:.3f}"
                )
        body.append(cells)

    n_variants = max(1, len(variants))
    fig, ax = plt.subplots(figsize=(7.1, max(4.0, 0.34 * (len(body) + 1) + 1.1)),
                           constrained_layout=True)
    ax.set_axis_off()
    ax.set_title(
        "Evaluator-level results (scores on native 0–1 scale; variability across judge passes)",
        loc="left", fontsize=7, fontweight="bold", pad=6,
    )
    first_width, type_width = 0.28, 0.09
    value_width = (1 - first_width - type_width) / n_variants
    table = ax.table(
        cellText=body,
        colLabels=["Evaluator", "Type", *variants],
        cellLoc="right",
        colLoc="right",
        colWidths=[first_width, type_width, *([value_width] * n_variants)],
        edges="closed",
        loc="center",
    )
    for row_i in range(len(body) + 1):
        table[(row_i, 0)].get_text().set_ha("left")
        table[(row_i, 1)].get_text().set_ha("left")
    _style_table(table)
    # Color is quantitative here: deeper blue indicates a larger continuous
    # score. Gates use teal/coral and retain explicit text for non-color readers.
    from matplotlib.colors import to_rgb

    def blend(low, high, amount):
        lo = to_rgb(low)
        hi = to_rgb(high)
        return tuple(a + (b - a) * amount for a, b in zip(lo, hi))

    for row_i, evaluator in enumerate(evaluators, start=1):
        sample = next(row for row in rows if row["evaluator"] == evaluator)
        is_gate = sample["evaluator_type"] == "gate"
        table[(row_i, 0)].set_facecolor(PALE_TEAL if is_gate else PALE_BLUE)
        table[(row_i, 1)].set_facecolor(PALE_TEAL if is_gate else PALE_BLUE)
        for column_i, variant in enumerate(variants, start=2):
            row = lookup.get((evaluator, variant))
            if not row or row["mean"] is None:
                continue
            if is_gate:
                color = blend(PALE_CORAL, PALE_TEAL, row["mean"])
            else:
                color = blend("#F5F8FA", "#8DBBD3", row["mean"])
            table[(row_i, column_i)].set_facecolor(color)
    fig.text(
        0.01, 0.01,
        "Values are mean ± sample SD across judge passes. Binary gates are shown as pass rates; "
        "they do not have continuous scores.",
        fontsize=5.5, ha="left", va="bottom",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_problems_table(path: Path, rows: list[dict]) -> None:
    plt = _mpl()
    variants = []
    problems = []
    scores = {}
    labels = {}
    for row in rows:
        if row["variant_id"] not in variants:
            variants.append(row["variant_id"])
            labels[row["variant_id"]] = row["plot_label"]
        if row["problem"] not in problems:
            problems.append(row["problem"])
        scores[(row["problem"], row["variant_id"])] = row["mean_score"]
    body = []
    for problem in problems:
        cells = [problem]
        for variant in variants:
            value = scores.get((problem, variant))
            cells.append("—" if value is None else f"{value:.1f}")
        body.append(cells)
    n_variants = max(1, len(variants))
    first_width = 0.34 if n_variants <= 4 else 0.28
    other_width = (1 - first_width) / n_variants
    fig, ax = plt.subplots(figsize=(7.1, max(1.25, 0.30 * (len(body) + 1))),
                           constrained_layout=True)
    ax.set_axis_off()
    table = ax.table(
        cellText=body,
        colLabels=["Problem", *(labels[v] for v in variants)],
        cellLoc="right",
        colLoc="right",
        colWidths=[first_width, *([other_width] * n_variants)],
        edges="closed",
        loc="center",
    )
    for row in range(len(body) + 1):
        table[(row, 0)].get_text().set_ha("left")
    _style_table(table)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _source_rows(records: list[Record]) -> list[dict]:
    return [{
        "judge": r.judge,
        "variant_id": r.variant.variant_id,
        "variant": r.variant.label,
        "replicate": r.replicate,
        "pass_index": r.pass_index,
        "problem": r.problem,
        "problem_id": r.problem_id,
        "composite_score": r.composite_score,
        "tier": r.tier,
        "compiled": r.compiles,
        "executed": r.ran,
        "gates_passed": r.gates_passed,
        "gates_total": r.gates_total,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "cached_tokens": r.cached_tokens,
        "total_tokens": r.total_tokens,
        "cost_usd": r.agent_cost_usd,
        "model_calls": r.model_calls,
        "tool_calls": r.tool_calls,
        "peak_context_tokens": r.peak_context_tokens,
        "execution_time_sec": r.execution_time_sec,
        "agent_wall_time_sec": r.agent_wall_time_sec,
        "response_replayed": r.response_replayed,
        "source_file": r.source_file,
    } for r in records]


def _safe_name(judge: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in judge).strip("-").lower()


def _build_slice(records: list[Record], out: Path, judge: str, pooled: bool) -> dict:
    out.mkdir(parents=True)
    n_problems = len({r.problem for r in records})
    enough_problems = n_problems >= MIN_FIGURE_PROBLEMS
    summary = _overview(records)
    evaluators = _evaluator_rows(records)
    problems = _matrix(records, out, render=enough_problems)
    _write_csv(out / "table_summary.csv", summary)
    _write_csv(out / "table_evaluators.csv", evaluators)
    _write_csv(out / "table_problems.csv", problems)
    _write_csv(out / "source_data_problem_runs.csv", _source_rows(records))
    _render_summary_table(out / "table_summary.pdf", summary)
    _render_evaluator_table(out / "table_evaluators.pdf", evaluators)
    if n_problems >= 2:
        _render_problems_table(out / "table_problems.pdf", problems)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "judge": judge,
        "pooled_judges": pooled,
        "n_problems": n_problems,
        "n_problem_runs": len(records),
        "problems": sorted({r.problem for r in records}),
        "variants": [v.variant_id for v in _variant_order(records)],
        "source_files": sorted({r.source_file for r in records}),
        "uncertainty": (
            "None reported. The problem set is fixed and curated, so a score over it is "
            "a measurement, not an estimate from a sample. Run-to-run spread needs "
            "repetitions; judge spread is reported as judge_score_sd once a run is rescored."
        ),
        "figure_policy": (
            "Figures emitted only with at least 3 observed problems; otherwise tables only."
        ),
    }
    (out / "included_results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _check_destination(dirs: list[Path] | None, destination: Path) -> None:
    # `build` swaps the bundle in by replacing `destination` outright, which
    # would delete the very results being read.
    for directory in dirs or SOURCE_DIRS:
        source = Path(directory).resolve()
        if source == destination or destination in source.parents:
            raise SystemExit(
                f"--out {destination} would replace the result directory {source}. "
                "Build the bundle somewhere outside the results."
            )


def build(dirs: list[Path] | None, destination: Path, include_pooled: bool = False) -> list[dict]:
    records = load_records(dirs, with_sources=False)
    if not records:
        raise SystemExit("No complete, provenanced result records found.")
    destination = destination.resolve()
    _check_destination(dirs, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    manifests = []
    try:
        by_judge: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            by_judge[record.judge].append(record)
        for judge in sorted(by_judge):
            manifests.append(_build_slice(by_judge[judge], temporary / _safe_name(judge), judge, False))
        if include_pooled and len(by_judge) > 1:
            manifests.append(_build_slice(records, temporary / "pooled", "all judges", True))
        backup_root = None
        backup = None
        if destination.exists():
            backup_root = Path(tempfile.mkdtemp(
                prefix=f".{destination.name}-previous-", dir=destination.parent))
            backup = backup_root / "bundle"
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup_root is not None and backup_root.exists():
            shutil.rmtree(backup_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, action="append",
                        help="result directory to scan; repeat for multiple repetitions")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="publication bundle directory (default: publication)")
    parser.add_argument("--pooled", action="store_true",
                        help="also emit an explicitly labelled cross-judge pooled view")
    args = parser.parse_args()
    manifests = build(args.dir, args.out, args.pooled)
    print(f"{args.out}: {len(manifests)} judge view(s), "
          f"{sum(m['n_problem_runs'] for m in manifests if not m['pooled_judges'])} problem-runs")


if __name__ == "__main__":
    main()
