# Dashboard work in progress

Handoff note, rewritten 2026-10-01 after the page was cut down to three cards.
Read this first when resuming.

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
| Detail | Every run behind the clicked cell, with the weighted metric strip, evaluator rows, and the source panel | unchanged from the old drill-down |
| All runs | The full per-record table, behind a toggle, hidden by default | unchanged |

**The previous page is kept verbatim as `dashboard_template_full.html`**, 89362
bytes. `analysis/` is untracked in git, so that file is the only copy. Do not
delete it without asking.

**`build_dashboard.py` was not touched.** The payload still carries `contrasts`,
`ci`, `variance`, `factors` and `factorLabels`; the page simply no longer reads
them. That kept the cut to one file. Trimming the payload is optional cleanup,
worth roughly 10 KB, and would mean rewriting the stats module's callers.

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
| `analysis/dashboard_template.html` | 849 lines. The three-card page. CSS plus markup plus one inline script. |
| `analysis/dashboard_template_full.html` | The eleven-card page, verbatim. Reference only, never built. |
| `analysis/load_results.py` | Unchanged. `SOURCE_DIRS` is the top level of `output/` only. |
| `analysis/build_dashboard.py` | Unchanged. Serializes with `allow_nan=False` through `_nulls_for_nan`. |
| `analysis/make_demo_runs.py` | Unchanged. Generates the six-variant synthetic set. |
| `analysis/devtools/domstub.js` | Minimal DOM shim. No `<select>` support, by omission. |
| `analysis/devtools/dash_run.js` | Static render check. Rewritten for the nine surviving ids. |
| `analysis/devtools/dash_drive.js` | Drives every control. Rewritten, grid replaces heatmap and the measure segment replaces colorBy. |
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
purple variant and a `judge_model`, and everything that passes is included.
There is no dedup, so the same run saved under two filenames counts twice, and
nothing reports what was skipped. Both are open.

**Open, and a data question rather than a page one. Two thin arms sit in the
comparison.** The simplification removed the cards that amplified them. It did
not remove them.

- `output/ext-smoke-judged-by-gpt52-run1.json`, landed 2026-10-01 15:30, loads
  as a fifth agent with 7 records, all scoring 0, six of seven failing to
  compile. It has full problem coverage, so nothing marks it thin.
- `single-pdesim-claudeopus481m-c3` has one record on one problem and **ranks
  first** on the scoreboard at 54.2, because one lucky run beats any honest
  average.

The page flags both as far as a page can. The coverage column is dimmed when it
is below 7 of 7, and the agent chip carries an amber `⚠ 1/7`. It still ranks
them by score, because inventing a minimum-n rule in the renderer would drop
data without saying so. The fix is upstream, either move the files out of the
top level or add the `--min-runs N` flag already on the list. **Asked the user
twice, no answer yet.**

## Verification status

Both sets fully clean, 2026-10-01 after the cut.

```
cd analysis && python build_dashboard.py
# 116 problem-runs, 5 variants, 1 run with source, 750 KB
node devtools/dash_run.js     # 1290 nodes in 16 ms, all 9 sections non-empty
node devtools/dash_drive.js   # all states clean, 16 table cols / 116 rows,
                              # 76 metric strips add up, source panel 101 lines
```

```
cd analysis && python make_demo_runs.py && python build_dashboard.py --demo
# 252 problem-runs, 6 variants, 192 runs with source, 866 KB
node devtools/dash_drive.js   # all states clean, 16 table cols / 252 rows,
                              # 267 metric strips add up, source panel 50 lines
```

The drive run exercises the judge filter in three states, all seven grid
measures, every agent chip hidden one at a time down to the last and then
restored, the table toggle, a grid cell selection, the source panel on open, and
the copy button. No card is permitted to empty in any state. Two behaviours make
that true rather than lucky. The last chip refuses to turn off, and the detail
panel re-homes to the first visible cell with runs when its own agent is hidden.

**Page contract, clean.** No doctype, html, head, or body tags. Verify with
`grep -oiE '<(!doctype|html|head|body)[ >]'`, NOT with `/<head/i`, which gives a
false positive on `<header>`. Title is inside the first 8 KB. Size 0.75 MB real
and 0.87 MB demo against the 16 MB limit.

## Repro commands

Run both after any template change. The demo set exercises the many-agent
geometry and the real set exercises the degenerate one, and neither covers the
other.

```bash
cd /scratch/hongzhang/petscagent_bench/analysis

# six agents, source panels populated
python make_demo_runs.py && python build_dashboard.py --demo && node devtools/dash_drive.js

# five agents, one with a single run, one source panel
python build_dashboard.py && node devtools/dash_drive.js
```

## Open, not started

- Dedup on `(variant, judge, run_index, problem)`.
- A build-time report naming every file that was skipped and why.
- A `--min-runs N` flag, which is also the clean fix for the two thin arms.

## Out of scope

- Stale output JSONs, the truncation bug, harness revision staleness banners.
  Closed by user instruction.
- Paper figures in `overleaf-paper/plot_results.py`. Not approved, do not start.
- Publishing as an artifact. Refused, because this session authenticates via
  `apiKeyHelper`, which takes precedence over a claude.ai login. Do not retry.
