# Dashboard work in progress

Handoff note, rewritten 2026-10-01 after the page was cut down to three cards
and updated 2026-10-02 after the branch was rebased onto main. Read this first
when resuming.

## What the user asked for

1. "Do not care about old output jsons. We will do new runs. Can you focus on
   visualization?" This is the controlling instruction. Nothing about stale
   JSONs, the truncation bug, or harness revision banners is in scope.
2. The dashboard must compare across purple agents, where future purples are
   things like multiagent, multiagent with skill, multiagent without skill.
   Each purple may be run multiple times.
3. Generated code for each problem must be reachable by clicking.
4. 2026-10-01, superseding most of what came before. "Looking at the current
   dashboard, I am very confused. There has been too much analysis which none
   cares about. Can you make the dashboard as simple as possible. I want it easy
   to check detailed info while gives high-level summary of results."

## The simplification, 2026-10-01

The page had grown to eleven cards. Nine of them answered a statistics question
nobody had asked. Eight were cut, on the user's explicit choice of "three cards"
over two softer options.

Gone: the contrast card with its paired bootstrap intervals, the strip plot, the
pipeline funnel, the judge matrix, the failure-mode breakdown, the category
matrix, the cost versus score Pareto, and the variance decomposition with its
projection curve. With them went the factor color model, the colorBy segment,
the three-slot categorical palette, the tooltip, and `fin(x)`.

What replaced them.

| Card | What it carries | What it absorbed |
| --- | --- | --- |
| Summary strip | Four numbers. Runs in view, share of submissions that executed, mean of scored runs, count reaching GOLD | the old KPI strip, cut from eight tiles to four |
| Agents | One row per agent. Coverage, submissions, compiled, ran, scored, then one mean column per judge | the funnel and the judge matrix, as columns |
| By problem | Agents by problems, one clickable cell each, under a measure segment of Score, Ran and the five metrics | the heatmap and the category matrix, behind one control |
| Detail | Every visible agent's runs on the selected problem, grouped by agent, each with the weighted metric strip, evaluator rows, and the source panel | the old drill-down, widened from a cell to a column on 2026-10-02 |
| All runs | The full per-record table, behind a toggle, hidden by default | unchanged |

**The previous page is kept verbatim as `dashboard_template_full.html`**, 89362
bytes. It is now tracked, committed in `44f6fb7` along with the rest of
`analysis/`, so history is a second copy. Do not delete it without asking.

**`build_dashboard.py` was not touched by the cut.** The payload still carries
`contrasts`, `ci`, `variance`, `factors` and `factorLabels`; the page simply no
longer reads them. That kept the cut to one file. Trimming the payload is
optional cleanup, worth roughly 10 KB, and would mean rewriting the stats
module's callers. The rebase pass below did touch the file, for the per-case
payload only.

## The rebase onto main, 2026-10-02

The branch was rebased onto main, which brought in per-test-case execution
(`154ed2b`) and the `--purple-url` launcher path (`86d4c94`). Neither broke the
build. Both changed what the page was telling the truth about, and three things
came out of it.

**A run now has `cases`.** Every declared `test_cases` entry runs with its own
`args` and `nsize`, and the result file records one entry per case. The loader
parses them into `Case` and `Record.cases`, the payload carries them trimmed,
and the detail panel renders a row strip in place of the old single argument
line. Only the failing cases carry stderr, capped at `CASE_STDERR_CHARS`,
because the passing ones are diagnosed by nothing and their PETSc chatter would
be most of the payload. A result file written before `154ed2b` has no `cases`,
loads with an empty list, and falls back to the argument line.

**There are now two kinds of zero.** A failing case no longer throws out of the
harness. The pipeline short-circuits on the failed gate, aggregation zeroes the
composite, and the record carries no error string. `Record.aborted` used to
reason that a zero with no error was impossible, which is why its docstring was
rewritten and `Record.gate_failed` added beside it. The panel marks the second
kind with its own banner, because a verdict of zero and a missing score look
identical in a number and should not look identical on the page.

**Provenance falls back to `reported_model`.** `launcher.py:111` fills
`purple_model` in only for a purple the green agent starts itself, so every run
against an already-running agent writes it empty. All three files currently in
`output/` are such runs and were being dropped at load with "No provenanced
result files found." They now load under the name the agent reported for itself,
at scaffold level `UNRECORDED`, which costs them their one-factor contrasts and
keeps the page from inventing a structure the file never recorded.
`pdesim-<model>-c<N>` is a composite, and reading it as a single agent would be
a claim nobody made.

The three files in `output/` predate the rebase, carry no `cases`, and are
there to test the visualization. The user will regenerate them against current
main. The per-case path is covered by the demo set until then.

## Design decisions that survive

**Mean score is the default grid cell**, chosen by the user over run count and
over a pass rate. Score means are taken over every run including the zeros, so
they sit well below the mean of scored runs in the summary strip. The card note
says so, because the gap between 68.6 and 41.4 is otherwise a mystery.

**Submissions, not records.** A submission is `(agent, problem, run)`. Two
judges score the same submission, so counting records would double every compile
and every execution. The scoreboard's compiled and ran columns dedup; the score
columns do not, because each judge's verdict is its own observation.

**One ordering for the whole page.** `shown()` returns visible agents sorted by
mean over the judges in view, best first, and the scoreboard rows, the grid rows
and the detail panel's default all read it. They used to disagree.

**Detail is scoped to the problem, not to the cell, 2026-10-02.** It used to
show the runs behind one cell, so comparing agents meant clicking each in turn
and remembering the last. It now lists every visible agent's runs on the
selected problem, grouped under a header carrying the agent, its run count and
its mean. The clicked agent leads and is underlined, the rest follow in the
page's one ordering, and the column header selects the problem with nobody
leading. Hiding an agent drops its group. On the demo set a column is six
groups and thirty-six collapsed runs, which is the cost of the comparison being
in one place.

**The grid selection is sticky, 2026-10-02.** A repeat click on the selected
cell used to clear it, and the detail panel fell back to its default landing on
the first agent. The user read that as the page refusing to show the other two
agents, which is fair, since the three names differ in one character and the
heading rendered that character at the same weight as everything else. The
repeat click now holds, the selected agent's row label goes bold alongside the
cell outline, and the heading leads with the agent name at 13px bold.

**Segmented buttons, never a `<select>`.** `domstub.js` has no `.value`, so a
dropdown is a control the headless drivers cannot exercise.

**The metric strip must add up on screen.** Contributions sum the displayed
figures, not the unrounded products, because a row that exists to be checked has
to survive being checked. `dash_drive.js` asserts this on every strip in every
cell. A missing category, or a composite this weighting does not reproduce, is
flagged in amber beside the total rather than hidden.

**Status palette is fixed, never themed, always icon plus label.** The
sequential blue ramp is now the only scale on the page, which is why the earlier
categorical palette work is no longer load bearing.

## Files

| File | State |
| --- | --- |
| `analysis/dashboard_template.html` | The three-card page. CSS plus markup plus one inline script. `caseStrip()` and the gate-failure banner are the rebase additions. |
| `analysis/dashboard_template_full.html` | The eleven-card page, verbatim. Reference only, never built. |
| `analysis/load_results.py` | `SOURCE_DIRS` is the top level of `output/` only. Carries `Case`, `Record.cases`, `Record.gate_failed`, and the `UNRECORDED` scaffold. |
| `analysis/build_dashboard.py` | Serializes with `allow_nan=False` through `_nulls_for_nan`. Emits `cases` and `gf` per row. |
| `analysis/make_demo_runs.py` | Generates the six-variant synthetic set. Evaluator names now match the real config, and failures split into compile failures and runtime case failures. |
| `analysis/devtools/domstub.js` | Minimal DOM shim. No `<select>` support, by omission. |
| `analysis/devtools/dash_run.js` | Static render check. Rewritten for the nine surviving ids. |
| `analysis/devtools/dash_drive.js` | Drives every control. Reads the judge segment's length rather than assuming two judges, because the real set has one. |
| `analysis/devtools/dash_dump.js` | Prints a section's visible text. Id agnostic, needed no change. `node devtools/dash_dump.js summary scoreboard` |

## Section ids

`prov`, `judgeseg`, `measureseg`, `tabletoggle`, `chiprow`, `summary`,
`scoreboard`, `gridnote`, `grid`, `ramp`, `detailctx`, `runlist`, `tablecard`,
`datatable`, `foot`. The old ids (`kpis`, `contrasts`, `heatmap`, `strip`,
`funnel`, `failmodes`, `judgematrix`, `catmatrix`, `pareto`, `variance`,
`projection`, `colorseg`, `lede`, `drillctx`) are gone. Anything still naming
one of them is stale.

## Where results come from

`SOURCE_DIRS` is the **top level of `output/` only**, set 2026-10-01 at the
user's instruction. Subdirectories are archives and side experiments and are
never scanned. `output/judge_swap` had been in the tuple and was silently mixing
a fourth run arm into every mean, which is why the record count dropped from 145
to 109 when it came out. `output/paper_v1` has no provenance and was already
excluded twice over.

Selection is otherwise a glob plus one gate. A file is kept if it yields both a
purple identity and a `judge_model`, and everything that passes is included.
Identity is `purple_model` when the green agent started the purple and
`reported_model` when it did not. There is no dedup, so the same run saved under
two filenames counts twice, and nothing reports what was skipped. Both are open.

`output/` currently holds three files, one per `pdesim-claudeopus481m-c{1,2,3}`
arm, each judged by `claudeopus46`, 21 records over 7 problems with one judge.
The thin arms that earlier versions of this note flagged, `ext-smoke` and the
one-record `c3`, are no longer in the top level. The page still flags thin
coverage the same way, a dimmed coverage column and an amber `⚠ n/7` on the
chip, and still ranks by score rather than inventing a minimum-n rule in the
renderer. The `--min-runs N` flag remains the clean fix and remains unbuilt.

## Verification status

Both sets fully clean, 2026-10-02 after the rebase pass.

```
cd analysis && python build_dashboard.py
# 21 problem-runs, 3 variants, 19 runs with source, 373 KB
node devtools/dash_run.js     # 1199 nodes in 18 ms, all 9 sections non-empty
node devtools/dash_drive.js   # all states clean, 16 table cols / 21 rows,
                              # 30 metric strips add up, source panel 115 lines
```

```
cd analysis && python make_demo_runs.py && python build_dashboard.py --demo
# 252 problem-runs, 6 variants, 199 runs with source, 927 KB
node devtools/dash_drive.js   # all states clean, 16 table cols / 252 rows,
                              # 278 metric strips add up, source panel 50 lines
```

The drive run exercises the judge filter in every state the data offers, all
seven grid measures, every agent chip hidden one at a time down to the last and
then restored, the table toggle, a grid cell selection, the source panel on
open, and the copy button. No card is permitted to empty in any state. Two
behaviours make that true rather than lucky. The last chip refuses to turn off,
and the detail panel re-homes to the first visible cell with runs when its own
agent is hidden.

The case strip and the gate-failure banner have no assertion of their own. The
real set cannot carry one, since those files predate `cases`, and the demo set
reaches both paths only through the detail panel the driver already opens.
Adding a check is worth doing once the regenerated runs land.

**Page contract, clean.** No doctype, html, head, or body tags. Verify with
`grep -oiE '<(!doctype|html|head|body)[ >]'`, NOT with `/<head/i`, which gives a
false positive on `<header>`. Title is inside the first 8 KB. Size 0.37 MB real
and 0.93 MB demo against the 16 MB limit.

## Repro commands

Run both after any template change. The demo set exercises the many-agent
geometry and the real set exercises the degenerate one, and neither covers the
other.

```bash
cd /scratch/hongzhang/petscagent_bench/analysis

# six agents, two judges, per-case strips and both kinds of zero
python make_demo_runs.py && python build_dashboard.py --demo && node devtools/dash_drive.js

# three agents, one judge, no cases recorded
python build_dashboard.py && node devtools/dash_drive.js
```

## Open, not started

- Dedup on `(variant, judge, run_index, problem)`.
- A build-time report naming every file that was skipped and why.
- A `--min-runs N` flag, the clean fix for a thin arm ranking first.
- A driver assertion on the case strip and the gate-failure banner.

## Out of scope

- Stale output JSONs, the truncation bug, harness revision staleness banners.
  Closed by user instruction.
- Paper figures in `overleaf-paper/plot_results.py`. Not approved, do not start.
- Publishing as an artifact. Refused, because this session authenticates via
  `apiKeyHelper`, which takes precedence over a claude.ai login. Do not retry.
