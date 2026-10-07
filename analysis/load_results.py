"""Load petscagent-bench result JSON into tidy records.

One record per (purple variant, judge, run index, problem). This is the single
source of truth for both the paper figures and the HTML dashboard, so that the
two can never disagree about what a number means.

The purple axis is a FACTOR MODEL, not a flat label. A variant is a point in
(scaffold, base_model, skills, n_subagents), and the interesting quantities are
contrasts along one factor rather than a ranking of the whole set. Everything
downstream -- the contrast plot, the facet choice, the color budget -- reads
those factors rather than parsing a display name.

Result files that carry no factor block are still loaded, under the flat model
tag the run recorded, so old and new runs sit on one axis. Their factors are
marked unrecorded rather than guessed, which costs those arms their one-factor
contrasts and is the honest price of not knowing.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

# Directories scanned for result files, in load order. Overridable so the
# dashboard can be built against a scratch directory without touching output/.
#
# Top level of output/ only, never a subdirectory. Subfolders there are archives
# and side experiments (output/paper_v1, output/judge_swap), and scanning them
# silently mixed an extra run arm into every mean. A result that belongs in the
# comparison goes at the top level; anything filed deeper is out by construction.
# A variant directory holds a copy of every score in its aggregate, one file per
# problem, so a recursive glob would load each record a second time as well.
# output-rep<N> holds a repetition of the grid. Each one regenerates the code,
# so they are separate runs of the same arm rather than extra scorings of one.
SOURCE_DIRS: tuple[Path, ...] = (OUTPUT_DIR, *sorted(REPO_ROOT.glob("output-rep*")))

# Display names for the model tags recorded in the result files.
MODEL_LABELS = {
    "anthropic/claudeopus46": "Claude Opus 4.6",
    "openai/gpt52": "GPT-5.2",
    "openai/gemini25pro": "Gemini 2.5 Pro",
}

# Scaffold level for a run whose configuration was never recorded. The green
# agent writes an empty `purple_model` whenever it evaluates an already-running
# agent, because `launcher.py` fills that tag in only for a purple it starts
# itself. Such a run is identified by the name the agent reported for itself,
# and a name does not say how the agent is built: `pdesim-<model>-c<N>` is a
# composite. Reading it as a single agent would be a claim the file never made.
UNRECORDED = "unrecorded"

SCAFFOLD_LABELS = {
    "single": "Single agent",
    "multiagent": "Multi-agent",
    UNRECORDED: "scaffold not recorded",
}

# Compact forms for axis ticks and heatmap headers, where the full label would
# wrap to three lines and squeeze the plot.
MODEL_SHORT = {
    "anthropic/claudeopus46": "Claude",
    "openai/gpt52": "GPT-5.2",
    "openai/gemini25pro": "Gemini",
}

SCAFFOLD_SHORT = {
    "single": "single",
    "multiagent": "multi",
    UNRECORDED: "scaffold ?",
}

# Categories and weights, mirrored from config/green_agent_config.yaml. Kept here
# so the dashboard can show the weighting without parsing YAML at build time.
# tests/test_scoring_weights.py holds this equal to the config, because a mirror
# that drifts shows a plausible weighting that no score was computed under.
CATEGORY_WEIGHTS = {
    "correctness": 0.35,
    "performance": 0.15,
    "code_quality": 0.15,
    "algorithm": 0.15,
    "petsc": 0.20,
}

# Which category each evaluator feeds. Read from the harness rather than copied,
# so the detail view groups rows exactly the way the score was aggregated. An
# evaluator missing from this map is a gate, which gates the composite but is
# never averaged into a category.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.metrics.aggregation import MetricsAggregator as _Aggregator  # noqa: E402

EVALUATOR_CATEGORY = dict(_Aggregator.EVALUATOR_CATEGORY_MAP)

TIER_ORDER = ("GOLD", "SILVER", "BRONZE", "FAIL")

# The factors a variant is built from. A contrast is a pair of variants that
# differ in exactly one of these, which is what makes its difference readable:
# with two factors moving at once the effect cannot be assigned to either.
FACTORS = ("scaffold", "base_model", "skills", "n_subagents")

FACTOR_LABELS = {
    "scaffold": "Scaffold",
    "base_model": "Base model",
    "skills": "Skills",
    "n_subagents": "Sub-agents",
}


def label_for(model_tag: str) -> str:
    """Human-readable name for a model tag, falling back to the tag itself."""
    return MODEL_LABELS.get(model_tag, model_tag)


@dataclass(frozen=True)
class Variant:
    """One purple agent configuration: a point in the factor space."""

    variant_id: str
    scaffold: str
    base_model: str
    skills: tuple[str, ...]
    n_subagents: int

    @property
    def label(self) -> str:
        """Short name for axes and legends.

        Built from the factors rather than stored, so two variants can never
        carry the same name while differing, or differ in name while identical.
        """
        # The scaffold is always named, including the control level. Dropping it
        # leaves the baseline reading as a bare model name, which in a drill-down
        # header is indistinguishable from the judge. The exception is a run that
        # never recorded one, where the name is all there is to print.
        parts = [label_for(self.base_model)]
        if self.scaffold != UNRECORDED:
            parts.append(SCAFFOLD_LABELS.get(self.scaffold, self.scaffold))
            if self.n_subagents > 1:
                parts[-1] += f" ×{self.n_subagents}"
        if self.skills:
            parts.append("+skills")
        return " · ".join(parts)

    def factor(self, name: str) -> Any:
        """Factor value by name, with skills flattened for comparison."""
        if name == "skills":
            return ",".join(self.skills) if self.skills else "none"
        return getattr(self, name)

    def factor_label(self, name: str) -> str:
        """Display form of one factor value."""
        value = self.factor(name)
        if name == "base_model":
            return label_for(value)
        if name == "scaffold":
            return SCAFFOLD_LABELS.get(value, value)
        if name == "skills":
            return "none" if value == "none" else value
        return str(value)

    def short_label(self, varying: Sequence[str]) -> str:
        """Axis-width name mentioning only the factors that actually vary.

        A label that repeats the same scaffold and model on every column costs
        horizontal space to say nothing, and pushes the real difference off the
        end. With one factor varying this collapses to just that factor's level.
        """
        bits = []
        for name in varying:
            if name == "base_model":
                bits.append(MODEL_SHORT.get(self.base_model, self.base_model.split("/")[-1]))
            elif name == "scaffold":
                bits.append(SCAFFOLD_SHORT.get(self.scaffold, self.scaffold))
            elif name == "n_subagents":
                bits.append(f"×{self.n_subagents}")
            elif name == "skills":
                bits.append("+skills" if self.skills else "no skills")
        return " ".join(bits) or self.label


def _parse_variant(doc: dict) -> Variant | None:
    """Read the purple configuration from a result document.

    Accepts the factor block when present. Failing that it takes a flat tag,
    preferring the configured `purple_model` and falling back to the
    `reported_model` the agent gave for itself, which is the only identity a
    run against an already-running purple carries. A flat tag names the arm but
    does not describe it, so the factors around it are left at `UNRECORDED`
    rather than invented; see the note on that constant.
    """
    purple = doc.get("purple")
    if isinstance(purple, dict) and purple.get("base_model"):
        skills = tuple(sorted(purple.get("skills") or ()))
        scaffold = purple.get("scaffold") or "single"
        n_sub = int(purple.get("n_subagents") or 1)
        vid = purple.get("variant_id") or "-".join(
            [scaffold, purple["base_model"], ",".join(skills) or "none", str(n_sub)]
        )
        return Variant(vid, scaffold, purple["base_model"], skills, n_sub)

    tag = doc.get("purple_model") or doc.get("reported_model")
    if not tag:
        return None
    return Variant(f"tag-{tag}", UNRECORDED, tag, (), 1)


@dataclass
class Evaluation:
    """One evaluator's verdict on one problem-run."""

    name: str
    type: str  # gate | metric | quality
    method: str
    passed: bool | None
    score: float | None
    confidence: float | None
    feedback: str


@dataclass
class SourceFile:
    """One generated C file, as written next to the result JSON."""

    filename: str
    sha256: str
    text: str


@dataclass
class Case:
    """One invocation of the compiled binary.

    A problem declares its own test cases and the harness runs each with its own
    arguments and rank count, so GradShafranov is six invocations rather than
    one. The execution gate fails the whole run when any single case fails,
    which makes a zero score unreadable unless the failing case can be named.

    Runs made before the harness ran every case carry no `cases` block and load
    with an empty list, which the dashboard reports as unrecorded rather than as
    a run with no invocations.
    """

    index: int
    args: str
    nsize: int
    runs: bool
    stderr: str
    execution_time_sec: float | None


@dataclass
class Record:
    """One problem attempted by one variant, under one judge, in one run."""

    variant: Variant
    judge: str
    # Which repetition of the arm this is. Repetitions generate their own
    # code, so they live in separate output directories and the directory is
    # the name. A legacy file numbered its repetitions inside one directory,
    # so its number is appended.
    replicate: str
    # Which scoring pass over those submissions, 0 for a legacy file that
    # recorded no pass at all.
    pass_index: int
    problem: str
    problem_id: str
    composite_score: float
    tier: str
    compiles: bool
    ran: bool
    gates_passed: int
    gates_total: int
    categories: dict[str, float]
    cli_args: str
    wall_time_sec: float
    execution_time_sec: float
    total_tokens: int | None
    harness_rev: str = ""
    petsc_rev: str = ""
    error: str = ""
    evaluations: list[Evaluation] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)
    sources: list[SourceFile] = field(default_factory=list)
    source_file: str = ""
    # How many files the submission itself recorded, read from the result
    # JSON. Distinct from `sources`, which is read from the run tree and is
    # empty whenever a caller loads with_sources=False, so only this one can
    # be asked whether the purple produced anything.
    submitted_sources: int = 0

    @property
    def agent(self) -> str:
        """Stable identity of the purple arm, for grouping."""
        return self.variant.variant_id

    @property
    def aborted(self) -> bool:
        """True when the run threw before any evaluator could score it.

        The harness caught an exception while building or running the code,
        wrote the message to `evaluation_summary.error`, and left
        `evaluation_details` null. The zero that follows is the absence of a
        verdict rather than one, which is why these must not be averaged in
        with real scores.
        """
        return bool(self.error)

    @property
    def gate_failed(self) -> bool:
        """True when the evaluators ran and a gate failed, scoring the run zero.

        The other way to score zero, and a different thing from `aborted`. A
        failing test case no longer propagates out of the harness: the case is
        recorded, the execution gate fails on it, and `aggregation.py`
        short-circuits the composite to zero with no error text. So this zero IS
        a verdict, reached on evidence that lives in `cases` and in the gate's
        feedback rather than in a stack dump.

        Runs made before the harness ran every declared case cannot be in this
        state, because a failing case threw instead of being scored.
        """
        return not self.error and self.gates_total > 0 and self.gates_passed < self.gates_total

    @property
    def failure_class(self) -> str:
        """Coarse failure mode for an aborted run, or "" when it completed.

        Compile status is checked first: a run that never built cannot have a
        meaningful runtime signature. Among runs that did build, a segfault is
        separated from an ordinary PETSc error because the two point at very
        different defects (memory handling versus API or solver misuse).
        """
        if not self.error:
            return ""
        # Nothing reached the compiler: the purple's reply held no code the
        # harness could extract. It arrives with compiles false like a real
        # compile failure, and calling it one blames the wrong stage and hides
        # the only failure mode the benchmark cannot blame on PETSc.
        #
        # The empty submission is what says so, not the message. A rescore of
        # such a record used to overwrite the parse error with one of its own,
        # and matching on text alone let those files fall through to the
        # compile branch and blame PETSc for a failure on the purple's side.
        if not self.submitted_sources or "parse purple agent response" in self.error:
            return "No code produced"
        if not self.compiles:
            return "Compile failure"
        if "SEGV" in self.error or "Segmentation Violation" in self.error:
            return "Segfault"
        if "PETSC ERROR" in self.error:
            return "PETSc runtime error"
        if "nonzero returncode" in self.error:
            return "Nonzero exit"
        return "Other"


# Fixed draw order for the failure-mode bars. The status role assigned to each
# mode does follow severity (compile failure warning, PETSc error serious,
# segfault critical), but the drawing order does not: severity order would put
# the warning and serious steps side by side, and that pair measures OKLab dE
# 13.6 for normal vision, under the 15 floor. The status palette is fixed and
# cannot be re-stepped, so the separation has to come from the order instead.
# This one clears the CVD and normal-vision gates on both surfaces; verify with
#   node scripts/validate_palette.js "#0ca30c,#fab219,#d03b3b,#ec835a" --mode light
# "No code produced" leads because it is a different stage rather than a
# severity step: it is the one mode where the harness never got as far as the
# compiler, so it sits outside the ordering the rest of this list encodes.
FAILURE_CLASSES = ("No code produced", "Compile failure", "Segfault",
                   "PETSc runtime error", "Nonzero exit", "Other")


def _result_files(dirs: Sequence[Path] | None = None) -> list[Path]:
    seen: list[Path] = []
    for directory in dirs or SOURCE_DIRS:
        if directory.is_dir():
            seen.extend(sorted(p for p in directory.glob("*.json") if p.is_file()))
    return seen


def _parse_evaluations(raw: Any) -> list[Evaluation]:
    # An aborted run has `evaluation_details` null rather than an empty list,
    # because nothing ever ran to produce entries. See `Record.aborted`.
    if not raw:
        return []
    out = []
    for e in raw:
        out.append(
            Evaluation(
                name=e.get("name", ""),
                type=e.get("type", ""),
                method=e.get("method", ""),
                passed=e.get("passed"),
                score=e.get("score"),
                confidence=e.get("confidence"),
                feedback=e.get("feedback") or "",
            )
        )
    return out


def _parse_cases(raw: Any) -> list[Case]:
    # Absent on every run made before the harness ran each declared case, and
    # on a run that never compiled. Both read as "no invocations recorded".
    if not raw:
        return []
    out = []
    for i, c in enumerate(raw):
        runtime = c.get("execution_time_sec")
        out.append(
            Case(
                index=int(c.get("index", i)),
                args=c.get("args") or "",
                nsize=int(c.get("nsize") or 1),
                runs=bool(c.get("runs")),
                stderr=c.get("stderr") or "",
                execution_time_sec=float(runtime) if runtime is not None else None,
            )
        )
    return out


def _display_path(path: Path) -> str:
    """Repo-relative path when the file lives in the repo, else the full path.

    Demo and scratch directories sit outside the repo, so a bare `relative_to`
    would raise on exactly the inputs used to check the layout.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_dirs(result_path: Path, doc: dict[str, Any]) -> list[Path]:
    """Where the code for `result_path` might be, best candidate first.

    The green agent writes the tree under the purple variant and names it in
    the document. Before that it wrote one tree per result file, under "runs"
    and earlier under "sources", so those are still read rather than losing
    the code of every run made on one side of a rename.

    The variant cut from the filename comes last. It is a guess, and in a
    directory holding both shapes it would attach a new variant's tree to an
    old aggregate and show the wrong code under its scores.
    """
    here = result_path.parent
    named = doc.get("submissions")
    guessed = result_path.stem.split("-judged-by-")[0]
    return [
        *([here / named] if named else []),
        here / "runs" / result_path.stem,
        here / "sources" / result_path.stem,
        *([here / guessed] if guessed != result_path.stem else []),
    ]


def _load_sources(
    result_path: Path, doc: dict[str, Any], max_bytes: int
) -> dict[str, list[SourceFile]]:
    """Read the generated C files the green agent saved for a result file.

    Returns a mapping from problem name to its files. Missing directories are
    normal: every run made before source persistence landed has none, and the
    dashboard says so rather than implying the code was empty.
    """
    for source_dir in _source_dirs(result_path, doc):
        manifest_path = source_dir / "manifest.json"
        if manifest_path.is_file():
            break
    else:
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}

    by_problem: dict[str, list[SourceFile]] = {}
    for entry in manifest:
        path = source_dir / entry.get("filename", "")
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            text = text[:max_bytes] + f"\n/* truncated at {max_bytes} bytes */\n"
        by_problem.setdefault(entry.get("problem_name", ""), []).append(
            SourceFile(
                filename=Path(entry.get("filename", "")).name,
                sha256=entry.get("sha256", ""),
                text=text,
            )
        )
    return by_problem


def _newest_passes(docs: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    """Keep the last scoring pass over each set of submissions.

    A rescore writes a new file rather than overwriting the one it replays,
    so without this every pass counts as another observation and a judge that
    was run twice gets twice the weight in every mean.

    Keyed per directory, because a repetition is a directory and its passes
    are its own. Filtered per file rather than per record, because the judge
    is read per record and a narrowed rescore carries records naming the
    previous judge, whose pass number is the new file's all the same.

    A document with no pass_index was written before passes were numbered.
    Those are always kept, or the three repetitions an older run wrote side
    by side in one directory would collapse into one.
    """
    def key(path: Path, doc: dict) -> tuple:
        variant = _parse_variant(doc)
        return (path.parent, variant.variant_id if variant else None,
                doc.get("judge_model"))

    newest: dict[tuple, int] = {}
    for path, doc in docs:
        if doc.get("pass_index") is not None:
            k = key(path, doc)
            newest[k] = max(newest.get(k, 0), int(doc["pass_index"]))
    return [
        (path, doc) for path, doc in docs
        if doc.get("pass_index") is None
        or int(doc["pass_index"]) == newest[key(path, doc)]
    ]


def load_records(
    dirs: Sequence[Path] | None = None,
    with_sources: bool = True,
    max_source_bytes: int = 60_000,
) -> list[Record]:
    """Read every provenanced result file into a flat list of records."""
    records: list[Record] = []
    docs = [(path, json.loads(path.read_text())) for path in _result_files(dirs)]
    for path, doc in _newest_passes(docs):
        variant = _parse_variant(doc)
        judge = doc.get("judge_model")
        if not variant or not judge:
            continue  # no provenance, cannot be placed on the judge axis
        # The document names the judge of the pass that last wrote it. A
        # narrowed rescore carries the problems it did not evaluate through
        # from the file it replays, so some records in the file were scored by
        # an earlier judge and say so in scored_by. Taking the judge per record
        # keeps those on their own axis instead of crediting them to the new
        # one. Records written before scored_by existed fall back to the
        # document, which for them is right.
        pass_index = int(doc.get("pass_index") or 0)
        # A legacy file numbered its repetitions within one directory, so the
        # directory alone would merge them. Everything written since puts a
        # repetition in its own directory and the name is enough.
        replicate = _display_path(path.parent)
        if doc.get("pass_index") is None:
            replicate += f"#run{doc.get('run_index', 0)}"
        harness = doc.get("harness") or {}
        env = doc.get("env") or {}
        sources = _load_sources(path, doc, max_source_bytes) if with_sources else {}

        for r in doc.get("results", []):
            summary = r.get("evaluation_summary") or {}
            problem = r.get("problem_name", "?")
            records.append(
                Record(
                    variant=variant,
                    judge=str(r.get("scored_by") or judge),
                    replicate=replicate,
                    pass_index=pass_index,
                    problem=problem,
                    problem_id=str(r.get("problem_id", "")),
                    composite_score=float(r.get("composite_score") or 0.0),
                    tier=r.get("tier") or "FAIL",
                    compiles=bool(r.get("compiles")),
                    ran=bool(r.get("runs")),
                    gates_passed=int(summary.get("gates_passed") or 0),
                    gates_total=int(summary.get("gates_total") or 0),
                    categories=dict(r.get("category_scores") or {}),
                    # A run with no arguments records a blank or a single
                    # space. Stripping it keeps the drill-down from showing an
                    # empty argument box that looks like missing data.
                    cli_args=(r.get("cli_args") or "").strip(),
                    wall_time_sec=float(r.get("time_used_sec") or 0.0),
                    execution_time_sec=float(r.get("execution_time_sec") or 0.0),
                    total_tokens=r.get("total_tokens"),
                    harness_rev=str(harness.get("git_rev") or ""),
                    petsc_rev=str(env.get("petsc_rev") or ""),
                    error=str(summary.get("error") or ""),
                    evaluations=_parse_evaluations(r.get("evaluation_details")),
                    cases=_parse_cases(r.get("cases")),
                    sources=sources.get(problem, []),
                    source_file=_display_path(path),
                    submitted_sources=len(r.get("generated_sources") or []),
                )
            )
    return records


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------


@dataclass
class Contrast:
    """A pair of variants differing in exactly one factor."""

    factor: str
    base: Variant  # the control level
    test: Variant  # the level being tested
    n_problems: int = 0

    @property
    def key(self) -> str:
        return f"{self.factor}:{self.base.variant_id}->{self.test.variant_id}"

    @property
    def label(self) -> str:
        return f"{self.test.factor_label(self.factor)} vs {self.base.factor_label(self.factor)}"

    @property
    def arms(self) -> str:
        """The two variants by name.

        The factor label alone is not unique: with both a 4-way and an 8-way
        multi-agent in the set, two scaffold contrasts read "Multi-agent vs
        Single agent" and hold the same factors. Naming the arms always
        separates them.
        """
        return f"{self.test.label}  vs  {self.base.label}"

    @property
    def held(self) -> str:
        """The factors held fixed, which is what makes the contrast readable.

        Only factors that genuinely match are listed. A nested factor moves with
        its parent (see `_differing`), so claiming it was held would be a false
        statement about what the contrast controls for.
        """
        other = [
            f for f in FACTORS
            if f != self.factor and self.base.factor(f) == self.test.factor(f)
        ]
        return ", ".join(f"{FACTOR_LABELS[f]} {self.base.factor_label(f)}" for f in other)


# Which level of each factor reads as the control, so a positive delta always
# means "the thing being tested helped". Anything not listed falls back to
# alphabetical order, which is arbitrary but at least stable.
CONTROL_LEVELS = {
    "scaffold": "single",
    "skills": "none",
}


def _differing(a: Variant, b: Variant) -> list[str]:
    """Factors separating two variants, after collapsing nested ones.

    `n_subagents` is nested inside `scaffold` rather than crossed with it: a
    single agent has exactly one worker by definition, so the 1 is structural
    and carries no information. Compared naively, every single-vs-multi pair
    moves two factors and gets thrown out, which would silently delete the
    scaffold contrast -- the main thing the suite is being extended to measure.
    So when the scaffold itself differs, the sub-agent count is not a second
    factor, it is part of what "multi-agent" means.
    """
    differing = [f for f in FACTORS if a.factor(f) != b.factor(f)]
    if "scaffold" in differing and "single" in (a.scaffold, b.scaffold):
        single, multi = (a, b) if a.scaffold == "single" else (b, a)
        if single.n_subagents == 1:
            differing = [f for f in differing if f != "n_subagents"]
    return differing


def find_contrasts(variants: Iterable[Variant]) -> list[Contrast]:
    """Enumerate every pair of variants separated by exactly one factor.

    Deriving these rather than listing them by hand means a new variant cannot
    be added without its comparisons appearing, and means no comparison can be
    drawn across two factors at once by accident.
    """
    out: list[Contrast] = []
    for a, b in combinations(sorted(variants, key=lambda v: v.variant_id), 2):
        differing = _differing(a, b)
        if len(differing) != 1:
            continue
        factor = differing[0]
        control = CONTROL_LEVELS.get(factor)
        if control is not None and b.factor(factor) == control:
            base, test = b, a
        elif control is not None and a.factor(factor) == control:
            base, test = a, b
        else:
            base, test = (a, b) if str(a.factor(factor)) < str(b.factor(factor)) else (b, a)
        out.append(Contrast(factor=factor, base=base, test=test))
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _per_problem(records: Iterable[Record], value: Callable[[Record], float]) -> dict[str, float]:
    by_problem: dict[str, list[float]] = {}
    for rec in records:
        by_problem.setdefault(rec.problem, []).append(value(rec))
    return {p: sum(v) / len(v) for p, v in by_problem.items()}


def bootstrap_ci(
    records: Iterable[Record],
    value: Callable[[Record], float],
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and 95% CI, resampling PROBLEMS rather than individual runs.

    The problem is the unit of generalization: a benchmark with six problems has
    six independent observations no matter how many times each is rerun. Treating
    each run as independent would shrink the interval by sqrt(runs) and overstate
    how well the suite separates the agents.

    Returns (mean, lo, hi); the interval is (nan, nan) when no records are given.
    """
    per_problem_map = _per_problem(records, value)
    if not per_problem_map:
        return (math.nan, math.nan, math.nan)

    per_problem = [per_problem_map[p] for p in sorted(per_problem_map)]
    mean = sum(per_problem) / len(per_problem)

    rng = random.Random(seed)
    n = len(per_problem)
    draws = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += per_problem[rng.randrange(n)]
        draws.append(total / n)
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    return (mean, lo, hi)


def paired_bootstrap(
    test: Iterable[Record],
    base: Iterable[Record],
    value: Callable[[Record], float],
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Mean paired difference (test - base) and its 95% CI.

    Pairing on problem cancels problem difficulty, so a six-problem suite can
    resolve a one-factor change that separate means cannot.

    How much it buys is an empirical question, not a given. On the current data
    the two components are comparable, 544 between problems against 899 within,
    so pairing is worth far less here than an earlier version of this docstring
    claimed. What it buys in any one contrast is `correlation`: at r 0.95 the
    paired interval is a third the width of the unpaired ones, at r near zero it
    is no better. Both show up in the dashboard, which is why r ships beside
    every contrast and a row with low r should be read as wide on purpose.

    Returns (mean_delta, lo, hi, n_paired_problems).
    """
    a = _per_problem(test, value)
    b = _per_problem(base, value)
    shared = sorted(set(a) & set(b))
    if not shared:
        return (math.nan, math.nan, math.nan, 0)

    deltas = [a[p] - b[p] for p in shared]
    mean = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    n = len(deltas)
    draws = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        draws.append(total / n)
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    return (mean, lo, hi, n)


def correlation(
    test: Iterable[Record],
    base: Iterable[Record],
    value: Callable[[Record], float],
) -> float:
    """Pearson r between the two arms' per-problem means.

    This is the diagnostic for whether pairing bought anything. Near +1 the
    paired interval collapses; near 0 pairing is no better than absolute and can
    be worse, because it adds the two arms' variances instead of cancelling them.
    """
    a = _per_problem(test, value)
    b = _per_problem(base, value)
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return math.nan
    xs = [a[p] for p in shared]
    ys = [b[p] for p in shared]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return math.nan
    return num / (dx * dy)


def variance_components(
    records: Iterable[Record],
    value: Callable[[Record], float],
) -> tuple[float, float]:
    """Between-problem and within-problem (run-to-run) variance.

    The ratio decides where the next unit of compute should go. When
    between-problem dominates, extra runs of the same problems buy almost
    nothing and extra problems buy a lot; the dashboard says which.
    """
    by_problem: dict[str, list[float]] = {}
    for rec in records:
        by_problem.setdefault(rec.problem, []).append(value(rec))
    if len(by_problem) < 2:
        return (math.nan, math.nan)

    means = {p: sum(v) / len(v) for p, v in by_problem.items()}
    grand = sum(means.values()) / len(means)
    between = sum((m - grand) ** 2 for m in means.values()) / (len(means) - 1)

    within_num = 0.0
    within_den = 0
    for p, vals in by_problem.items():
        if len(vals) < 2:
            continue
        m = means[p]
        within_num += sum((v - m) ** 2 for v in vals)
        within_den += len(vals) - 1
    within = within_num / within_den if within_den else 0.0
    return (between, within)
