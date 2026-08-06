#!/usr/bin/env python3
"""Drive the full benchmark grid on Argo: 3 models x N repetitions x M judges.

The model strings and endpoints below are specific to ANL's Argo gateway.
Adapt MODELS to run the same grid against another provider.

Model string and api_base_url are coupled on Argo, so they are always set
together from MODELS below. Argo exposes two endpoints:

  /argoapi/v1  OpenAI-compatible. GPT and Gemini live here (they 404 on the
               bare path). Claude also works here but caps near 20k tokens.
  /argoapi     Anthropic-native. Only Claude works here; accepts 32k tokens.

Claude therefore uses the native endpoint so that max_tokens=32000 is usable
without truncation.

Usage:
    uv run run_argo_grid.py --judges gpt52 claudeopus46 --reps 3
    uv run run_argo_grid.py --judges gpt52 --reps 1 --dry-run
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

V1 = "https://apps.inside.anl.gov/argoapi/v1"
NATIVE = "https://apps.inside.anl.gov/argoapi"

# slug -> (litellm model string, api_base_url)
MODELS = {
    "claudeopus46": ("anthropic/claudeopus46", NATIVE),
    "gemini25pro": ("openai/gemini25pro", V1),
    "gpt52": ("openai/gpt52", V1),
}

PURPLE_CFG = Path("config/purple_agent_config.yaml")
GREEN_CFG = Path("config/green_agent_config.yaml")
OUTPUT_DIR = Path("output")


def set_yaml_field(path, key, value):
    """Replace `key: ...` in a YAML file, preserving indentation and comments."""
    text = path.read_text()
    pattern = re.compile(rf'^(\s*){key}:\s*"?[^"\n#]*"?(\s*#.*)?$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise SystemExit(f"could not find '{key}:' in {path}")
    if len(matches) > 1:
        raise SystemExit(f"'{key}:' is ambiguous in {path} ({len(matches)} matches)")
    m = matches[0]
    indent, comment = m.group(1), (m.group(2) or "")
    text = text[: m.start()] + f'{indent}{key}: "{value}"{comment}' + text[m.end() :]
    path.write_text(text)


def configure(purple_slug, judge_slug):
    pm, pu = MODELS[purple_slug]
    jm, ju = MODELS[judge_slug]
    set_yaml_field(PURPLE_CFG, "model", pm)
    set_yaml_field(PURPLE_CFG, "api_base_url", pu)
    set_yaml_field(GREEN_CFG, "model", jm)
    set_yaml_field(GREEN_CFG, "api_base_url", ju)
    return pm, pu, jm, ju


def existing_outputs():
    return {p.name for p in OUTPUT_DIR.glob("*.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["gpt52"], choices=list(MODELS))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir", default="grid_logs")
    args = ap.parse_args()

    # Preserve the configs so an interrupted grid does not leave them mangled.
    backups = {p: p.with_suffix(p.suffix + ".gridbak") for p in (PURPLE_CFG, GREEN_CFG)}
    for src, dst in backups.items():
        shutil.copy2(src, dst)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    plan = [
        (j, m, r)
        for j in args.judges
        for m in args.models
        for r in range(1, args.reps + 1)
    ]
    print(f"grid: {len(plan)} runs "
          f"({len(args.models)} models x {args.reps} reps x {len(args.judges)} judges)\n")

    results = []
    try:
        for i, (judge, model, rep) in enumerate(plan, 1):
            pm, pu, jm, ju = configure(model, judge)
            label = f"[{i}/{len(plan)}] {model} judged-by {judge} rep{rep}"
            print(f"{label}\n    purple={pm} @ {pu}\n    green ={jm} @ {ju}")
            if args.dry_run:
                results.append((label, "DRY-RUN", None))
                continue

            before = existing_outputs()
            t0 = time.time()
            log = log_dir / f"{model}-judge-{judge}-rep{rep}.log"
            with open(log, "w") as fh:
                rc = subprocess.call(["uv", "run", "main.py", "launch"],
                                     stdout=fh, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            new = sorted(existing_outputs() - before)
            status = "OK" if (rc == 0 and new) else f"FAILED(rc={rc})"
            print(f"    -> {status} in {dt:.0f}s  new={new or 'NONE'}  log={log}\n")
            results.append((label, status, new[0] if new else None))
    finally:
        for src, dst in backups.items():
            shutil.copy2(dst, src)
            dst.unlink()
        print("configs restored")

    print("\n=== summary ===")
    for label, status, out in results:
        print(f"  {status:12s} {label}  {out or ''}")
    bad = [r for r in results if r[1].startswith("FAILED")]
    if bad:
        print(f"\n{len(bad)} run(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
