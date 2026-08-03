"""Purple Agent implementation - the target code generation agent being tested.

This Purple Agent is responsible for generating PETSc C/C++ code from natural
language problem descriptions. It operates as an A2A-compliant agent that:

1. Receives problem descriptions via the A2A protocol
2. Uses an LLM to generate PETSc code
3. Returns generated code files along with CLI arguments
4. Maintains conversation context for multi-turn interactions

The agent is isolated from the evaluation logic and is the subject of testing
by the Green Agent through the petscagent-bench framework. The Green Agent can
grade v2 without any changes.
"""

import argparse
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
import dotenv
import litellm
from litellm import completion
from pydantic import BaseModel
from loguru import logger

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities, TextPart, FilePart, FileWithBytes
from a2a.utils import new_agent_parts_message

import petscmcp
from petsc_compile_run_mcp_client import PetscCompileRunMCPClient

# Optional RAG import. rag_retrieve(query) does a vector search over the
# pre-built PETSc corpus and returns top-K chunks; rag_format(chunks) turns
# them into a prompt-ready string. If src.petsc_rag isn't on this machine,
# both fall back to None and downstream code skips RAG via `is not None`.
try:
    from src.petsc_rag.retrieve import retrieve as rag_retrieve, format_for_prompt as rag_format
except Exception:
    rag_retrieve = None
    rag_format = None

# Optional header-signature lookup. Given text, _hdr_extract finds PETSc
# symbol names (KSPSolve, MatCreate, ...); _hdr_lookup returns the exact C
# signature parsed from PETSc .h files; _hdr_format builds a prompt block;
# _hdr_size reports index size for status logs. Prevents "wrong-signature"
# LLM mistakes (e.g. "gmres" vs KSPGMRES) before compile ever sees them.
try:
    from src.petsc_rag.headers import (
        extract_petsc_symbols as _hdr_extract,
        format_signature_block as _hdr_format,
        size as _hdr_size,
        lookup as _hdr_lookup,
    )
except Exception:
    _hdr_extract = None
    _hdr_format = None
    _hdr_size = None
    _hdr_lookup = None

# Load .env so PETSC_DIR, LLM API keys, and MCP_SERVER_URL are available
# via os.environ.get(...) throughout the rest of the file.
dotenv.load_dotenv()


# System prompt that defines the code generation contract for v2.
# The strict JSON output rule lets downstream code json.loads the response
# directly; the PETSc guidance blocks compress common failure modes; the
# ITERATION rule tells the LLM how to behave on self-verify fix-it turns.
SYSTEM_CODE_CONTRACT = (
    "You are a PETSc code-generation agent. You produce compilable, runnable PETSc C/C++/CUDA programs.\n"
    "\n"
    "OUTPUT FORMAT (strict):\n"
    "Return ONLY a single raw JSON object. No markdown, no backticks, no code blocks, no explanation outside the JSON.\n"
    "Top-level JSON keys MUST be exactly: 'codes', 'nsize', 'cli_args' (no additional keys).\n"
    "- 'codes': list of {'filename', 'code'} objects. Code strings must be valid C/C++/CUDA.\n"
    "- 'nsize': number of MPI processes (use 1 for sequential).\n"
    "- 'cli_args': command-line argument string (e.g. '-ts_type beuler -ts_monitor').\n"
    "- First file in 'codes' is the main file.\n"
    "- Any explanations MUST live inside C block comments /* ... */ within the code strings.\n"
    "\n"
    "PETSc GUIDANCE:\n"
    "- Always start with PetscInitialize / PetscFinalize and check return codes with PetscCall.\n"
    "- Pick the right solver family for the problem: TS for time-dependent ODE/PDE, SNES for nonlinear systems,\n"
    "  KSP for linear systems, TAO for optimization, DM (DMDA/DMPlex) for structured/unstructured meshes.\n"
    "- Vec/Mat objects must be created, sized, assembled (VecAssemblyBegin/End, MatAssemblyBegin/End) and\n"
    "  destroyed before PetscFinalize.\n"
    "- Prefer command-line tunability: expose method choices via options (-ts_type, -snes_type, -ksp_type, -pc_type)\n"
    "  rather than hard-coding when the problem leaves it open.\n"
    "- Compile against $PETSC_DIR/$PETSC_ARCH with the standard makefile pattern; the build environment provides\n"
    "  a generic GNUmakefile that uses `make <executable>` and links PETSc automatically.\n"
    "\n"
    "ITERATION:\n"
    "- If the user provides a compile error or runtime error message from a previous attempt, fix the SPECIFIC\n"
    "  problem (missing header, wrong signature, mismatched type, undefined symbol) and regenerate the full\n"
    "  JSON. Do not abbreviate or use '...'.\n"
)


def load_purple_agent_v2_config(config_path: str = "config/purple_agent_v2_config.yaml") -> Dict[str, Any]:
    """Load purple agent v2 configuration from file or use defaults.

    Supports both JSON and YAML formats. Format is auto-detected by file extension.
    The returned dict shape also documents the full v2 configuration surface —
    any knob not present here isn't a real knob.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                # Detect format by extension
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            print(f"@@@ Purple agent v2: loaded config from {config_path}")
            return data
        except Exception as e:
            print(f"@@@ Purple agent v2: failed to load config {config_path}: {e}. Using defaults.")
    else:
        print(f"@@@ Purple agent v2: config {config_path} not found, using defaults")
    # Fall back to default configuration
    return {
        'llm': {'model': 'anthropic/claude-opus-4-5', 'api_base_url': None, 'temperature': 0.0},
        'mcp': {'server_url': os.environ.get('MCP_SERVER_URL', 'http://localhost:8080/mcp')},
        'self_fix': {
            'max_iters': 3,
            'do_smoke_run': True,
            'smoke_run_timeout_sec': 20,
            'check_diagnostics': True,
            'diagnostic_flags': [
                '-ksp_converged_reason',
                '-snes_converged_reason',
                '-ts_monitor',
            ],
        },
        'rag': {'enabled': False, 'k': 4, 'rerank': False, 'k_initial': 12},
        'header_lookup': {'enabled': True, 'limit': 8},
    }


def prepare_purple_agent_v2_card(url: str) -> AgentCard:
    """Create an A2A agent card for the Purple Agent v2.

    The agent card is a metadata descriptor that advertises the agent's
    capabilities, skills, and communication modes to potential clients.
    Wire-compatible with the baseline (v1) purple agent's card shape so
    the Green Agent grades both variants identically.

    Args:
        url: The base URL where this agent is accessible

    Returns:
        AgentCard object describing the Purple Agent v2's capabilities
    """
    skill = AgentSkill(
        id="petsc_code_generation_v2",
        name="PETSc Code Generation (v2, self-verifying)",
        description=(
            "Generates PETSc C/C++/CUDA code from natural language problem descriptions, "
            "then compiles and smoke-runs the result via the PETSc compile-run MCP server "
            "before returning. Wire-compatible with the baseline purple agent."
        ),
        tags=["purple agent", "code generation", "PETSc", "HPC", "self-verifying"],
        examples=["Write a PETSc program that solves the Robertson ODE system using TS."],
    )
    return AgentCard(
        name="purple_agent_v2",
        description=(
            "Self-verifying PETSc code generation agent. Same A2A protocol as the baseline "
            "purple agent, but compiles and (optionally) runs its own output via MCP before "
            "replying to the Green Agent."
        ),
        url=url,
        version="0.2.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/octet-stream"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


# Regex for turning an arbitrary A2A context_id into a filesystem-safe basename.
_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_]+')


# Diagnostic regexes for parsing PETSc runtime output. Each targets a specific
# failure signature that indicates the program ran but produced garbage.
_DIAG_DIVERGED = re.compile(r'\bDIVERGED_\w+', re.IGNORECASE)
_DIAG_NAN_INF = re.compile(r'(?:Floating point exception|\b[+-]?(?:nan|inf)\b)', re.IGNORECASE)
_DIAG_PETSC_ERROR = re.compile(r'\[\d+\]PETSC ERROR:')

# Intentionally NOT a fix-triggering signature: `Option left: name:-foo` has very
# low precision — it fires on perfectly correct programs that pass a flag only
# one of several solver objects consumes. Treating it as a failure misled the
# LLM into corrupting otherwise-working code (see diag_ab_results: Robertson
# -14.1, Rosenbrock -2.6, scatter_vecmpi -6.8 with this enabled). Keep the flag
# out of the default `diagnostic_flags` list so PETSc doesn't even print it.


# Human-readable hints appended to the fix-it turn per diagnostic kind.
# Gives the LLM a starting point beyond just the raw stderr.
_DIAG_HINTS = {
    "divergence": (
        "This usually means the wrong solver/preconditioner for this matrix, an ill-conditioned "
        "or non-symmetric assembly, or missing/incorrect boundary conditions. Reconsider the "
        "solver chain (KSP/SNES/TS type, PC type) and the matrix assembly."
    ),
    "nan_inf": (
        "NaN or Inf indicates uninitialized memory, division by zero, or a malformed nonlinear "
        "residual/Jacobian. Check Vec/Mat assembly, initial conditions, and any user-supplied "
        "function/Jacobian routines."
    ),
    "petsc_error": (
        "PETSc printed an internal error even though the process didn't crash. Read the error "
        "stack and fix the underlying call (wrong object type, missing setup step, "
        "incompatible sizes, etc.)."
    ),
}


def _parse_petsc_diagnostics(stdout: str, stderr: str = "") -> dict | None:
    """Scan PETSc diagnostic output for known failure signatures.

    Returns None if nothing failed, or a dict {kind, matches, output_tail} otherwise.
    Priority order: divergence > nan_inf > petsc_error > options_left (options_left is
    the weakest signal so it only fires alone).
    """
    combined = f"{stdout}\n{stderr}" if stderr else stdout
    if not combined:
        return None

    lines = combined.splitlines()
    # Highest-priority: solver explicitly diverged.
    diverged_lines = [ln for ln in lines if _DIAG_DIVERGED.search(ln)]
    if diverged_lines:
        return {
            "kind": "divergence",
            "matches": diverged_lines[:5],
            "output_tail": "\n".join(lines[-30:]),
        }

    # Next: numerical blow-up (NaN / Inf / floating-point exception).
    nan_lines = [ln for ln in lines if _DIAG_NAN_INF.search(ln)]
    if nan_lines:
        return {
            "kind": "nan_inf",
            "matches": nan_lines[:5],
            "output_tail": "\n".join(lines[-30:]),
        }

    # Lowest-priority: PETSc printed an internal error block.
    petsc_err = _DIAG_PETSC_ERROR.findall(combined)
    if petsc_err:
        idx = combined.find("PETSC ERROR:")
        block = combined[max(0, idx - 80):idx + 1500] if idx >= 0 else combined[-1500:]
        return {"kind": "petsc_error", "matches": petsc_err[:3], "output_tail": block}

    return None


def _format_diagnostic_failure(diag: dict) -> str:
    """Render a parser dict as the body of a fix-it user turn."""
    hint = _DIAG_HINTS.get(diag["kind"], "")
    matches_block = "\n  ".join(diag["matches"])
    return (
        f"The previous code ran without crashing, but PETSc diagnostics reported a "
        f"`{diag['kind']}` failure:\n\n"
        f"  {matches_block}\n\n"
        f"Relevant output tail:\n```\n{diag['output_tail']}\n```\n\n"
        f"{hint}\n"
        "Fix the underlying issue and return the full JSON object again."
    )


def _safe_basename(context_id: str | None, fallback: str = "v2prog") -> str:
    """Derive a filesystem-safe executable basename from the A2A context id.

    Each problem flows through Green with a distinct context_id (set to the
    problem name on the Green side). Sanitizing it gives the v2 agent a stable
    per-problem prefix on the MCP server, isolating one problem's build
    artifacts from another's.
    """
    if not context_id:
        return fallback
    cleaned = _SAFE_NAME_RE.sub('_', str(context_id)).strip('_')
    return (cleaned or fallback)[:40]


class CodeFile(BaseModel):
    """A single source file in the LLM's response.

    Attributes:
        filename: Filename the LLM chose (e.g. "main.c"); v2 renames the first
            entry to a per-problem basename before uploading to the MCP server.
        code: Raw C/C++/CUDA source as a string.
    """

    filename: str
    code: str


class ProblemResponse(BaseModel):
    """Structured schema for the LLM's code-generation response.

    Passed as `response_format` to litellm so the model is forced to produce
    parseable JSON matching this shape. Downstream code uses these fields
    directly without re-validating.

    Attributes:
        codes: One or more source files. First entry is the main file.
        nsize: Number of MPI processes to launch (1 = sequential).
        cli_args: Runtime flags passed to the compiled binary.
    """

    codes: list[CodeFile]
    nsize: int
    cli_args: str


class NumericalPlan(BaseModel):
    """Structured numerical-analysis plan committed to before any code is written.

    Mirrors the Numerical Analysis specialist's outputs in the larger DOE
    project: discretization choice, solver stack, the PETSc APIs the plan
    relies on (validated against installed headers), and how the program
    will verify itself at runtime.
    """

    problem_class: str
    discretization: str
    solver_stack: str
    key_petsc_apis: list[str]
    verification_strategy: str
    rationale: str


# System prompt for the plan-generation LLM call (plan-then-code feature).
# The plan locks the implementation to a discretization + solver + API surface
# BEFORE any C code is written, reducing mid-generation drift.
SYSTEM_PLAN_CONTRACT = (
    "You are a PETSc Numerical Analysis specialist. Before any code is written you produce a "
    "structured plan that commits the implementer to a discretization, a solver stack, and the "
    "specific PETSc APIs that will be used.\n"
    "\n"
    "OUTPUT FORMAT (strict):\n"
    "Return ONLY a single raw JSON object. No markdown, no backticks, no prose outside the JSON.\n"
    "Top-level keys MUST be exactly: 'problem_class', 'discretization', 'solver_stack', "
    "'key_petsc_apis', 'verification_strategy', 'rationale'.\n"
    "- 'problem_class': one short phrase (e.g. 'time-dependent stiff ODE', 'steady linear elliptic PDE on structured grid').\n"
    "- 'discretization': method + mesh family (e.g. 'cell-centered finite volume on DMDA 2D structured grid').\n"
    "- 'solver_stack': the TS/SNES/KSP/PC chain you intend to use (e.g. 'TSARKIMEX with SNES, KSP=GMRES, PC=ASM/ILU').\n"
    "- 'key_petsc_apis': list of the specific PETSc public function names the implementation will call "
    "(e.g. ['DMDACreate2d', 'KSPSolve', 'TSSetType']). 6-15 names. Names must be real PETSc symbols.\n"
    "- 'verification_strategy': how the program will check it produced a sensible answer at runtime "
    "without an oracle (residual norm, convergence-reason flag, conservation check, MMS-style, etc.).\n"
    "- 'rationale': 2-4 sentences justifying the discretization and solver_stack choices for this problem.\n"
    "\n"
    "GUIDANCE:\n"
    "- Prefer the simplest discretization that captures the physics; do not pick a fancier method unless the\n"
    "  problem requires it (stiff -> implicit; nonlinear -> SNES; eigenvalues -> SLEPc).\n"
    "- The solver_stack should be a defensible default an expert would write, not an exotic combination.\n"
    "- key_petsc_apis must be the names you actually intend to call from C. Prefer canonical create/setup/\n"
    "  setfromoptions/solve/destroy sequences over auxiliary helpers.\n"
)


# Prompt-varied plan generators for the multi-plan judge pattern. All run at
# temperature=0 (reproducible per-user-per-prompt); diversity comes from these
# static angle directives, NOT from sampling. Each addendum is appended to the
# common SYSTEM_PLAN_CONTRACT so the output schema stays identical and the
# judge can compare apples to apples.
PLAN_VARIANT_PROMPTS = [
    (
        "simplest",
        "ANGLE FOR THIS PLAN: pick the SIMPLEST discretization and solver stack that could possibly work. "
        "Bias toward the textbook teaching example. No limiters, no high-order schemes, no fancy preconditioners. "
        "Optimize for lines-of-code and reader comprehension.",
    ),
    (
        "accurate",
        "ANGLE FOR THIS PLAN: pick the discretization and solver stack with the HIGHEST EXPECTED ACCURACY for "
        "this problem's class. Prefer high-order schemes (RK4/SSP-RK, TVD/MUSCL limiters, P2/P3 finite elements, "
        "spectral when applicable), conservative formulations, and tight tolerances. Optimize for getting the "
        "right answer over getting it fast.",
    ),
    (
        "robust",
        "ANGLE FOR THIS PLAN: pick the discretization and solver stack that is MOST ROBUST to ill-conditioning, "
        "stiffness, and bad initial data. Prefer implicit schemes, stabilized formulations, mesh-refinement-aware "
        "preconditioners (GAMG, ASM), and L-stable time integrators (TSBDF, TSARKIMEX, TSROSW). Optimize for the "
        "code running to completion across a range of inputs.",
    ),
]


# System prompt for the plan-judge LLM call. Given N candidate plans for the
# same problem, scores each on a 5-criterion rubric and picks a winner.
SYSTEM_PLAN_JUDGE = (
    "You are a PETSc Numerical Analysis review panel. You receive several candidate plans for the SAME problem "
    "and must pick the ONE plan that is most likely to produce a correct, runnable PETSc program for the user's "
    "actual problem description.\n"
    "\n"
    "Score each plan on a 0-10 rubric over these criteria:\n"
    "  1. problem_fit: does the discretization match the physics (e.g. is upwinding used for hyperbolic, implicit for stiff)?\n"
    "  2. solver_appropriateness: is the solver stack defensible for this problem class?\n"
    "  3. api_consistency: do key_petsc_apis form a coherent program (create/setup/solve/destroy + the right object kinds)?\n"
    "  4. expected_accuracy: how close is the plan likely to get to the true answer the user wants?\n"
    "  5. expected_robustness: will the plan run to completion under reasonable inputs without divergence/NaN?\n"
    "\n"
    "OUTPUT FORMAT (strict):\n"
    "Return ONLY a single raw JSON object with these exact keys:\n"
    "  'winner_index': 0-based index of the chosen plan\n"
    "  'scores': list of {'index', 'total', 'notes'} per candidate, total is sum across the 5 criteria (max 50)\n"
    "  'rationale': 2-4 sentences explaining the winner choice\n"
    "\n"
    "GUIDANCE: bias slightly toward accuracy over simplicity — a simpler plan that gets a clearly worse answer "
    "is not preferable. But penalize unnecessary complexity that adds no accuracy or robustness benefit.\n"
)


class PlanScore(BaseModel):
    """One judge score entry for a single candidate plan.

    Attributes:
        index: 0-based position of the scored plan in the candidate list.
        total: Rubric sum across the 5 criteria (max 50).
        notes: Free-text justification for the score.
    """

    index: int
    total: int
    notes: str


class PlanJudgement(BaseModel):
    """Full judge response for a multi-plan comparison.

    Attributes:
        winner_index: 0-based index of the plan the judge picked.
        scores: Per-candidate rubric scores (may disagree with winner_index —
            see _select_plan for the reconciliation policy).
        rationale: 2-4 sentence explanation of the winner choice.
    """

    winner_index: int
    scores: list[PlanScore]
    rationale: str


def _validate_plan_apis(apis: list[str]) -> tuple[list[str], list[str], str]:
    """Filter plan-declared APIs against the installed PETSc header index.

    Returns (kept, dropped, signature_block) where:
      - kept     = APIs found in the index, signature_block lists their canonical signatures
      - dropped  = APIs the plan named that don't exist in the installed headers
      - signature_block = formatted hint text for the code-gen prompt (empty if none kept)

    Uses the same headers.py index the reactive header_lookup uses, but
    proactively (before any compile failure). Dropping nonexistent symbols
    is the proactive check that catches an LLM that hallucinated an API in
    the plan itself.
    """
    if not apis or _hdr_lookup is None or _hdr_format is None:
        return ([], list(apis or []), "")
    kept: list[str] = []
    dropped: list[str] = []
    for name in apis:
        if not isinstance(name, str) or not name.strip():
            continue
        n = name.strip()
        if _hdr_lookup(n) is not None:
            if n not in kept:
                kept.append(n)
        else:
            if n not in dropped:
                dropped.append(n)
    block = _hdr_format(kept) if kept else ""
    return (kept, dropped, block)


def _format_plan_for_prompt(plan: dict, sig_block: str, dropped: list[str]) -> str:
    """Render the locked plan + validated signatures as a code-gen system addendum."""
    pieces = [
        "LOCKED NUMERICAL PLAN (do not deviate without an explicit fix-it instruction):",
        f"  problem_class: {plan.get('problem_class', '?')}",
        f"  discretization: {plan.get('discretization', '?')}",
        f"  solver_stack: {plan.get('solver_stack', '?')}",
        f"  verification_strategy: {plan.get('verification_strategy', '?')}",
        f"  rationale: {plan.get('rationale', '?')}",
        "",
        "Implement against THIS plan. If the plan is unworkable, return a JSON object with a "
        "single key 'plan_unworkable' explaining why, instead of code.",
    ]
    if sig_block:
        pieces.append("")
        pieces.append(sig_block)
    if dropped:
        pieces.append("")
        pieces.append(
            "The plan named these symbols that DO NOT exist in the installed PETSc headers; "
            "do not call them (substitute the real API):\n  " + ", ".join(dropped)
        )
    return "\n".join(pieces)


def _format_plan_for_green(plan: dict, kept: list[str], dropped: list[str]) -> str:
    """Render the plan as a labeled block to append to the A2A TextPart.

    Green's parser is line-based on the existing nsize/cli_args/self_verify_*
    keys; an additional block below those lines is ignored by the parser and
    available to LLM-judge evaluators that read the full TextPart.
    """
    return (
        "numerical_plan:\n"
        f"  problem_class: {plan.get('problem_class', '?')}\n"
        f"  discretization: {plan.get('discretization', '?')}\n"
        f"  solver_stack: {plan.get('solver_stack', '?')}\n"
        f"  verification_strategy: {plan.get('verification_strategy', '?')}\n"
        f"  rationale: {plan.get('rationale', '?')}\n"
        f"  apis_validated: {', '.join(kept) if kept else '(none)'}\n"
        f"  apis_dropped: {', '.join(dropped) if dropped else '(none)'}\n"
    )


class PetscAgentV2Executor(AgentExecutor):
    """A2A executor with an inner compile/run self-verification loop.

    Extends the v1 executor pattern with:
      - Self-verification: after generation, upload+compile+smoke-run via MCP;
        feed failures back to the LLM as fix-it turns (up to max_iters).
      - RAG: inject retrieved PETSc tutorials into the first-turn system prompt.
      - Header lookup: inject canonical PETSc signatures on compile-error fix-it turns.
      - Plan-then-code: force a structured numerical plan (optionally multi-plan
        + judge) before writing any code; revise the plan after repeated failures.

    All features are config-gated; with everything off, behavior is close to v1.

    Attributes:
        model: LLM model identifier (e.g. "anthropic/claude-opus-4-5").
        ctx_id_to_messages: Dict mapping A2A context IDs to per-problem message
            history for multi-turn (fix-it) LLM conversations.
        ctx_id_to_plan: Dict mapping context IDs to the locked plan state
            (plan dict + validated/dropped API lists + revised flag).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor from a purple-agent-v2 config dict.

        Args:
            config: Config dict, typically loaded from
                config/purple_agent_v2_config.yaml via load_purple_agent_v2_config.
        """
        self.config = config
        # --- LLM settings ---
        llm_cfg = config.get('llm', {})
        self.model = llm_cfg.get('model')
        self.temperature = float(llm_cfg.get('temperature', 0))
        self.api_base_url = llm_cfg.get('api_base_url')

        # --- MCP compile-run server ---
        mcp_cfg = config.get('mcp', {})
        self.mcp_server_url = mcp_cfg.get('server_url') or os.environ.get('MCP_SERVER_URL', 'http://localhost:8080/mcp')

        # --- Self-fix / self-verify loop knobs ---
        sf = config.get('self_fix', {})
        self.max_iters = int(sf.get('max_iters', 3))
        self.do_smoke_run = bool(sf.get('do_smoke_run', True))
        self.smoke_run_timeout_sec = float(sf.get('smoke_run_timeout_sec', 20))
        self.check_diagnostics = bool(sf.get('check_diagnostics', True))
        # Wall-clock budget for a single problem's self-verify loop, in seconds.
        # Must be strictly less than Green's A2A read timeout (currently 3000s in
        # src/util/a2a_comm.py) so Purple always returns before Green kills the
        # request. Default 2400s = 40min leaves a 10-min safety margin.
        self.wall_clock_budget_sec = float(sf.get('wall_clock_budget_sec', 2400.0))
        # After plan revision fires, allow this many additional self-verify
        # attempts against the revised plan. Previously revision could fire on
        # attempt N and the loop would exit at attempt N+1 (or immediately),
        # meaning the revised plan was never actually tested. Small budget to
        # avoid blowing the wall-clock limit above.
        self.plan_revision_extra_iters = int(sf.get('plan_revision_extra_iters', 2))
        self.diagnostic_flags = list(sf.get('diagnostic_flags') or [
            '-ksp_converged_reason',
            '-snes_converged_reason',
            '-ts_monitor',
        ])

        # --- RAG (retrieval-augmented generation) ---
        rag_cfg = config.get('rag', {})
        self.rag_enabled = bool(rag_cfg.get('enabled', False))
        self.rag_k = int(rag_cfg.get('k', 4))
        self.rag_rerank = bool(rag_cfg.get('rerank', False))
        self.rag_k_initial = int(rag_cfg.get('k_initial', 2 * self.rag_k + 4))
        # Guard: config says enabled but the package didn't import (e.g. faiss
        # missing). Silently disable rather than crash on first request.
        if self.rag_enabled and rag_retrieve is None:
            print("@@@ Purple agent v2: rag.enabled=true but petsc_rag import failed; disabling RAG.")
            self.rag_enabled = False

        # --- Header signature lookup ---
        hdr_cfg = config.get('header_lookup', {})
        self.header_lookup_enabled = bool(hdr_cfg.get('enabled', True))
        self.header_lookup_limit = int(hdr_cfg.get('limit', 8))
        if self.header_lookup_enabled and _hdr_extract is None:
            print("@@@ Purple agent v2: header_lookup.enabled=true but headers module import failed; disabling.")
            self.header_lookup_enabled = False
        elif self.header_lookup_enabled:
            # Eager index-load so first-request latency isn't hit by a cold start.
            try:
                n = _hdr_size() if _hdr_size else 0
                print(f"@@@ Purple agent v2: header signature index loaded ({n} symbols)")
            except Exception as e:
                print(f"@@@ Purple agent v2: header_lookup load failed ({e}); disabling.")
                self.header_lookup_enabled = False

        # --- Plan-then-code feature ---
        plan_cfg = config.get('plan', {})
        self.plan_enabled = bool(plan_cfg.get('enabled', False))
        self.plan_allow_revision = bool(plan_cfg.get('allow_plan_revision', True))
        self.plan_emit_to_green = bool(plan_cfg.get('emit_to_green', True))
        self.plan_num_plans = int(plan_cfg.get('num_plans', 1))  # 1 = single-plan legacy; >1 = multi-plan + judge
        # Clamp num_plans to the number of angle variants we actually have.
        if self.plan_num_plans > len(PLAN_VARIANT_PROMPTS):
            print(
                f"@@@ Purple agent v2: plan.num_plans={self.plan_num_plans} exceeds "
                f"{len(PLAN_VARIANT_PROMPTS)} available variants; clamping."
            )
            self.plan_num_plans = len(PLAN_VARIANT_PROMPTS)
        if self.plan_num_plans < 1:
            self.plan_num_plans = 1
        if self.plan_enabled and _hdr_lookup is None:
            print("@@@ Purple agent v2: plan.enabled=true but headers module unavailable; API validation will be a no-op.")

        # Per-context conversation history for multi-turn (fix-it) interactions.
        self.ctx_id_to_messages: Dict[str, list] = {}
        # Per-context plan state: {context_id: {"plan": dict, "kept": list, "dropped": list, "revised": bool}}.
        self.ctx_id_to_plan: Dict[str, Dict[str, Any]] = {}

    def _llm_json_call(self, messages: list, response_format) -> Dict[str, Any]:
        """LLM call that returns a JSON-decoded dict. Shared by plan/judge paths."""
        completion_kwargs: Dict[str, Any] = {
            'messages': messages,
            'model': self.model,
            'temperature': self.temperature,
            'response_format': response_format,
            'timeout': 300,
        }
        litellm.ssl_verify = False
        # AskSage endpoints need explicit SSL cert + API key from env.
        if self.api_base_url:
            completion_kwargs['api_base'] = self.api_base_url
            if self.api_base_url.startswith('https://api.asksage.anl.gov'):
                litellm.ssl_verify = os.environ["ASKSAGE_SSL_CERT_FILE"]
                completion_kwargs['api_key'] = os.environ["ASKSAGE_API_KEY"]
        response = completion(**completion_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError(f"Expected string content from LLM, got {type(content)}")
        # Strip markdown fences some LLMs (e.g. Claude) add.
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            content = content.lstrip("json").strip()
        return json.loads(content)

    def _generate_plan(
        self,
        user_input: str,
        failure_history: str = "",
        variant_addendum: str = "",
    ) -> Dict[str, Any]:
        """One-shot LLM call that emits a structured NumericalPlan as a dict.

        variant_addendum is an angle directive (simplest/accurate/robust) appended
        to the common system prompt for multi-plan diversity. Empty for legacy
        single-plan callers.

        If failure_history is non-empty, it's appended as a user turn to push the
        revision away from whatever the prior plan led to.
        """
        sys_content = SYSTEM_PLAN_CONTRACT
        if variant_addendum:
            sys_content = f"{SYSTEM_PLAN_CONTRACT}\n\n{variant_addendum}"
        plan_msgs: list = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_input},
        ]
        if failure_history:
            plan_msgs.append({
                "role": "user",
                "content": (
                    "A previous plan for this same problem led to the following self-verify "
                    f"failures across multiple attempts:\n{failure_history}\n"
                    "Produce a REVISED plan that addresses the root cause. Choose a different "
                    "discretization, solver_stack, or verification_strategy as needed."
                ),
            })
        return self._llm_json_call(plan_msgs, NumericalPlan)

    def _generate_plan_set(self, user_input: str, failure_history: str = "") -> Dict[str, Any]:
        """Generate plan_num_plans candidate plans and pick one via the judge.

        Returns the winning plan dict. When num_plans=1, this is a single call
        with no variant_addendum and no judge invocation — byte-identical to
        the legacy single-plan path.
        """
        if self.plan_num_plans <= 1:
            return self._generate_plan(user_input, failure_history=failure_history)
        candidates: list[tuple[str, Dict[str, Any]]] = []
        for i, (label, addendum) in enumerate(PLAN_VARIANT_PROMPTS[: self.plan_num_plans]):
            if i > 0:
                time.sleep(0.5)
            try:
                p = self._generate_plan(
                    user_input,
                    failure_history=failure_history,
                    variant_addendum=addendum,
                )
                candidates.append((label, p))
            except Exception as e:
                print(f"@@@ Purple agent v2: plan variant '{label}' failed ({e}); skipping")
        if not candidates:
            raise RuntimeError("all plan variants failed; cannot proceed with plan-then-code")
        # Only one variant survived — skip the judge, no comparison to do.
        if len(candidates) == 1:
            label, plan = candidates[0]
            print(f"@@@ Purple agent v2: only 1 plan variant succeeded ('{label}'); skipping judge")
            return plan
        winner_label, winner_plan, judgement = self._select_plan(user_input, candidates)
        print(
            f"@@@ Purple agent v2: judge picked '{winner_label}' from {len(candidates)} variants "
            f"(scores: " + ", ".join(
                f"{candidates[s.index][0]}={s.total}" for s in judgement
                if 0 <= s.index < len(candidates)
            ) + ")"
        )
        return winner_plan

    def _select_plan(
        self, user_input: str, candidates: list[tuple[str, Dict[str, Any]]]
    ) -> tuple[str, Dict[str, Any], list[PlanScore]]:
        """Judge LLM picks the winner from candidate plans. Returns (label, plan, scores).

        Selection policy: rubric scores are authoritative. Pick argmax(score.total)
        across valid, in-range scores. Ties broken by `accurate > robust > simplest`
        (matches the SYSTEM_PLAN_JUDGE prompt's "bias slightly toward accuracy"
        guidance). Fall back to the LLM's `winner_index` only when scores are
        absent/malformed, and to candidate 0 when even that is out of range.

        Motivation: `winner_index` and `scores` are two independent LLM outputs
        from the same call and can disagree.
        """
        # Build the judge's user turn: the problem + every candidate plan.
        cand_block_lines = [f"USER PROBLEM:\n{user_input}\n", "CANDIDATE PLANS:"]
        for i, (label, p) in enumerate(candidates):
            cand_block_lines.append(
                f"\n--- candidate {i} (angle: {label}) ---\n{json.dumps(p, indent=2)}"
            )
        judge_msgs = [
            {"role": "system", "content": SYSTEM_PLAN_JUDGE},
            {"role": "user", "content": "\n".join(cand_block_lines)},
        ]
        # Preference order for tie-breaks; earlier = preferred.
        tiebreak_priority = {"accurate": 0, "robust": 1, "simplest": 2}
        try:
            raw = self._llm_json_call(judge_msgs, PlanJudgement)
            scores_raw = raw.get("scores", []) or []
            scores = [PlanScore(**s) for s in scores_raw if isinstance(s, dict)]
            valid_scores = [s for s in scores if 0 <= s.index < len(candidates)]
            if valid_scores:
                def sort_key(s: PlanScore) -> tuple[int, int]:
                    label = candidates[s.index][0]
                    return (-s.total, tiebreak_priority.get(label, 99))
                winner = sorted(valid_scores, key=sort_key)[0]
                idx = winner.index
                raw_idx = int(raw.get("winner_index", -1))
                # Judge disagreed with its own rubric — log and trust the rubric.
                if raw_idx != idx:
                    print(
                        f"@@@ Purple agent v2: judge self-inconsistency — winner_index="
                        f"{raw_idx} ('{candidates[raw_idx][0] if 0 <= raw_idx < len(candidates) else '?'}') "
                        f"but rubric argmax={idx} ('{candidates[idx][0]}'); using rubric"
                    )
                return candidates[idx][0], candidates[idx][1], scores
            # Fallback: no valid scores → trust winner_index.
            idx = int(raw.get("winner_index", 0))
            if 0 <= idx < len(candidates):
                print(f"@@@ Purple agent v2: judge returned no valid scores; using winner_index={idx}")
                return candidates[idx][0], candidates[idx][1], scores
            print(f"@@@ Purple agent v2: judge returned out-of-range index {idx}; using candidate 0")
        except Exception as e:
            print(f"@@@ Purple agent v2: judge call failed ({e}); using candidate 0")
        return candidates[0][0], candidates[0][1], []

    def _header_hint_block(self, error_text: str) -> str:
        """Look up canonical signatures for any PETSc symbols named in error_text.

        Called on compile-error and runtime-error fix-it turns. Returns a
        prompt-ready block of "here's the real signature" hints, or empty
        string if the feature is off / no symbols were found / lookup failed.
        """
        if not self.header_lookup_enabled or not error_text:
            return ""
        try:
            syms = _hdr_extract(error_text, limit=self.header_lookup_limit)  # type: ignore[misc]
            if not syms:
                return ""
            block = _hdr_format(syms)  # type: ignore[misc]
            if block:
                print(f"@@@ Purple agent v2: header lookup injected signatures for {syms}")
            return ("\n\n" + block) if block else ""
        except Exception as e:
            print(f"@@@ Purple agent v2: header lookup failed ({e}); skipping signature hint")
            return ""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a code generation request.

        This method is called by the A2A framework when a message is received.
        It processes the request through the following steps:

        1. Extract user input from the request context
        2. On first turn: build the system prompt (optionally with RAG-retrieved
           tutorials and a locked numerical plan) and initialize conversation history
        3. Call _generate_and_self_verify to drive the generate → compile → run
           → fix-it loop until a passing (or last-attempt) result is returned
        4. Assemble the A2A response (TextPart with metadata + one FilePart per
           generated code file) and enqueue it

        Args:
            context: Request context containing user input and metadata (including
                context_id, used for per-problem state and executable naming)
            event_queue: Queue for sending response events back to the client

        Note:
            The method handles both success and error cases, always returning
            a properly formatted A2A response.
        """
        user_input = context.get_user_input()

        # First turn for this context: build the initial system prompt.
        if context.context_id not in self.ctx_id_to_messages:
            system_content = SYSTEM_CODE_CONTRACT
            # Optionally splice retrieved PETSc tutorials into the system prompt.
            if self.rag_enabled:
                try:
                    hits = rag_retrieve(
                        user_input,
                        k=self.rag_k,
                        rerank=self.rag_rerank,
                        k_initial=self.rag_k_initial if self.rag_rerank else None,
                    )
                    ref_block = rag_format(hits)
                    if ref_block:
                        system_content = f"{SYSTEM_CODE_CONTRACT}\n\n{ref_block}"
                        mode = (
                            f"rerank({self.rag_k_initial}->{self.rag_k})"
                            if self.rag_rerank else f"vector(k={self.rag_k})"
                        )
                        print(
                            f"@@@ Purple agent v2: RAG[{mode}] injected {len(hits)} tutorial(s): "
                            + ", ".join(h['path'] for h in hits)
                        )
                except Exception as e:
                    print(f"@@@ Purple agent v2: RAG retrieval failed ({e}); falling back to base prompt.")
            # Optionally generate + lock a numerical plan before any code is written.
            if self.plan_enabled:
                try:
                    plan = self._generate_plan_set(user_input)
                    kept, dropped, sig_block = _validate_plan_apis(plan.get('key_petsc_apis', []) or [])
                    self.ctx_id_to_plan[context.context_id] = {
                        "plan": plan, "kept": kept, "dropped": dropped, "revised": False,
                    }
                    plan_block = _format_plan_for_prompt(plan, sig_block, dropped)
                    system_content = f"{system_content}\n\n{plan_block}"
                    print(
                        f"@@@ Purple agent v2: plan generated "
                        f"(discretization='{plan.get('discretization', '?')[:60]}', "
                        f"apis_kept={len(kept)}, apis_dropped={len(dropped)})"
                    )
                except Exception as e:
                    print(f"@@@ Purple agent v2: plan generation failed ({e}); falling back to no-plan path.")
            self.ctx_id_to_messages[context.context_id] = [{
                "role": "system",
                "content": system_content,
            }]
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})

        # Per-problem prefix so concurrent problems don't collide on the MCP server.
        basename = _safe_basename(context.context_id)

        try:
            data = await self._generate_and_self_verify(messages, basename, context.context_id, user_input)
        except Exception as e:
            # Any hard failure escaping the self-verify loop: report to Green as an error part.
            print(f"@@@ Purple agent v2: ❌ failed: {e}")
            await event_queue.enqueue_event(
                new_agent_parts_message(
                    [TextPart(text=f"Code generation failed ❌\nerror: {e}\n")],
                    context_id=context.context_id,
                )
            )
            return

        # Build the success response: a TextPart with metadata + one FilePart per source file.
        nsize = data["nsize"]
        cli_args = data["cli_args"]
        sv_attempts = data.get("self_verify_attempts", "?")
        sv_status = data.get("self_verify_status", "?")
        text_block = (
            f"Code generation successful ✅\n"
            f"nsize: {nsize}\n"
            f"cli_args: {cli_args}\n"
            f"self_verify_attempts: {sv_attempts}\n"
            f"self_verify_status: {sv_status}\n"
        )
        # Optionally append the plan as a labeled block for Green's LLM-judge evaluators.
        if self.plan_enabled and self.plan_emit_to_green:
            pstate = self.ctx_id_to_plan.get(context.context_id)
            if pstate is not None:
                text_block += "\n" + _format_plan_for_green(
                    pstate["plan"], pstate["kept"], pstate["dropped"]
                )
        parts_list: list = [TextPart(text=text_block)]
        for entry in data["codes"]:
            fwb = FileWithBytes(
                name=entry["filename"],
                bytes=entry["code"].encode("utf-8"),
                mime_type="text/plain",
            )
            parts_list.append(FilePart(file=fwb))
        await event_queue.enqueue_event(
            new_agent_parts_message(parts_list, context_id=context.context_id)
        )

    def _call_llm(self, messages: list) -> Dict[str, Any]:
        """LLM call that returns the code-generation response as a dict."""
        completion_kwargs: Dict[str, Any] = {
            'messages': messages,
            'model': self.model,
            'temperature': self.temperature,
            'response_format': ProblemResponse,
            'timeout': 300,
        }
        litellm.ssl_verify = False
        if self.api_base_url:
            completion_kwargs['api_base'] = self.api_base_url
            if self.api_base_url.startswith('https://api.asksage.anl.gov'):
                litellm.ssl_verify = os.environ["ASKSAGE_SSL_CERT_FILE"]
                completion_kwargs['api_key'] = os.environ["ASKSAGE_API_KEY"]
        response = completion(**completion_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError(f"Expected string content from LLM, got {type(content)}")
        # Strip markdown fences some LLMs add.
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            content = content.lstrip("json").strip()
        return json.loads(content)

    async def _generate_and_self_verify(
        self,
        messages: list,
        basename: str,
        context_id: str | None = None,
        user_input: str | None = None,
    ) -> Dict[str, Any]:
        """Generate code, then compile (and optionally smoke-run) it via MCP.

        On compile or run failure, append the stderr to `messages` as a user
        turn and ask the LLM to fix it. Returns the first response whose code
        passes the configured checks, or the last attempt if the loop is
        exhausted (so Green still gets *something* to grade and can apply its
        own outer retry).

        When plan_enabled + plan_allow_revision, two consecutive same-kind
        failures trigger a one-shot plan revision: the system prompt is
        rebuilt with a revised plan and the next attempt codes against it.
        Capped at one revision per problem to prevent thrashing.
        """
        last_data: Dict[str, Any] | None = None
        last_failure_kind: str | None = None
        last_failure_excerpt: str = ""
        attempts_used = 0
        failure_history: list[tuple[str, str]] = []  # (kind, excerpt_tail) per failed attempt
        t0 = time.monotonic()
        # Mutable cap so plan revision can grant a small extra allowance for the
        # revised plan to actually be tested. Only bumped once per problem.
        iter_cap = self.max_iters
        revision_extra_granted = False

        # Only catch MCP client setup failures here — anything that happens
        # *inside* a single attempt should be handled per-attempt so we burn
        # a retry on it instead of shipping broken code.
        try:
            mcp_ctx = PetscCompileRunMCPClient(self.mcp_server_url)
            mcp_client = await mcp_ctx.__aenter__()
            await mcp_client.initialize()
        except Exception as e:
            print(f"@@@ Purple agent v2: MCP setup failed ({e}); returning best-effort code")
            data = self._call_llm(messages)
            data["self_verify_attempts"] = 0
            data["self_verify_status"] = f"mcp_setup_failed: {e}"
            return data

        try:
            attempt = 0
            while attempt < iter_cap:
                attempt += 1
                attempts_used = attempt

                # Wall-clock guard: abort the loop if we're within striking distance
                # of Green's A2A timeout. Returning stale-but-parseable data is
                # strictly better than letting Green kill the request with a 0.
                elapsed = time.monotonic() - t0
                if elapsed > self.wall_clock_budget_sec:
                    print(
                        f"@@@ Purple agent v2: wall-clock budget exceeded "
                        f"({elapsed:.0f}s > {self.wall_clock_budget_sec:.0f}s) for {basename}; "
                        f"aborting loop with last attempt (failure: {last_failure_kind})"
                    )
                    break

                print(f"@@@ Purple agent v2: self-verify attempt {attempt}/{iter_cap} for {basename}")

                # Record the previous iteration's failure (if any) for plan-revision logic.
                if attempt > 1 and last_failure_kind is not None:
                    failure_history.append((last_failure_kind, last_failure_excerpt))

                # Plan-revision check: two consecutive failures of the same kind
                # and we haven't revised yet — regenerate the plan with the
                # failure history and replace the system prompt in-place.
                if (
                    self.plan_enabled
                    and self.plan_allow_revision
                    and context_id is not None
                    and user_input is not None
                    and len(failure_history) >= 2
                    and failure_history[-1][0] == failure_history[-2][0]
                ):
                    pstate = self.ctx_id_to_plan.get(context_id)
                    if pstate is not None and not pstate.get("revised"):
                        history_str = "\n".join(
                            f"  attempt {i+1}: {k} -- {ex[:300]}"
                            for i, (k, ex) in enumerate(failure_history)
                        )
                        try:
                            # Grant a small extra iteration budget so the revised
                            # plan actually gets tested. Without this, revision
                            # firing on the final loop iteration exits before any
                            # code against the revised plan is generated.
                            if not revision_extra_granted:
                                iter_cap += self.plan_revision_extra_iters
                                revision_extra_granted = True
                            print(
                                f"@@@ Purple agent v2: revising plan after {len(failure_history)} "
                                f"consecutive `{failure_history[-1][0]}` failures "
                                f"(iter_cap {self.max_iters}->{iter_cap})"
                            )
                            new_plan = self._generate_plan_set(user_input, failure_history=history_str)
                            kept, dropped, sig_block = _validate_plan_apis(
                                new_plan.get('key_petsc_apis', []) or []
                            )
                            self.ctx_id_to_plan[context_id] = {
                                "plan": new_plan, "kept": kept, "dropped": dropped, "revised": True,
                            }
                            plan_block = _format_plan_for_prompt(new_plan, sig_block, dropped)
                            # Rewrite the system prompt in-place to swap in the revised plan.
                            if messages and messages[0].get("role") == "system":
                                base_sys = messages[0]["content"].split("\n\nLOCKED NUMERICAL PLAN")[0]
                                messages[0]["content"] = f"{base_sys}\n\n{plan_block}"
                            messages.append({
                                "role": "user",
                                "content": (
                                    "The plan has been revised based on the failures above. "
                                    "Implement the REVISED plan now and return the full JSON object."
                                ),
                            })
                        except Exception as e:
                            print(f"@@@ Purple agent v2: plan revision failed ({e}); continuing with prior plan.")
                # Generate this attempt's code.
                try:
                    data = self._call_llm(messages)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    last_failure_kind = "llm_response_parse"
                    last_failure_excerpt = str(e)
                    print(f"@@@ Purple agent v2: LLM response was not valid JSON ({e}); feeding back for retry")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed as JSON ({e}). "
                            "Return ONLY a single JSON object matching the requested schema, "
                            "with no prose, code fences, or extra text after the closing brace."
                        ),
                    })
                    continue
                # Record the assistant turn so subsequent fix-it prompts see prior code.
                messages.append({"role": "assistant", "content": json.dumps(data)})
                last_data = data

                # Stage the files on the MCP server. We rename the first file
                # to {basename}.<ext> so `make {basename}` builds it; the rest
                # keep their names so headers/extra TUs are found.
                main_exec_name, upload_failure = await self._upload_files(mcp_client, data["codes"], basename)
                if upload_failure:
                    last_failure_kind = "upload"
                    last_failure_excerpt = upload_failure
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The previous response failed to stage on the build server:\n{upload_failure}\n"
                            "Regenerate the full JSON object with valid filenames and content."
                        ),
                    })
                    continue

                # Compile.
                try:
                    await mcp_client.make(executable=main_exec_name, dependencies="")
                    compile_ok = True
                    compile_stderr = ""
                except petscmcp.MCPDynamicClientReturnCode as e:
                    compile_ok = False
                    compile_stderr = (e.stderr or e.stdout or "")[-3500:]
                except petscmcp.MCPDynamicClientException as e:
                    compile_ok = False
                    compile_stderr = f"MCP error during make: {e}"

                if not compile_ok:
                    last_failure_kind = "compilation"
                    last_failure_excerpt = compile_stderr
                    # Enrich the fix-it turn with canonical signatures for any PETSc symbols in the error.
                    hint = self._header_hint_block(compile_stderr)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The previous code failed to compile. Compiler stderr (tail):\n"
                            f"```\n{compile_stderr}\n```\n"
                            "Fix the specific compilation errors and return the full JSON object again."
                            f"{hint}"
                        ),
                    })
                    continue

                # Optional smoke run.
                if not self.do_smoke_run:
                    data["self_verify_attempts"] = attempts_used
                    data["self_verify_status"] = "compile_ok_no_smoke"
                    return data

                # Inject diagnostic flags so we can parse solver output for silent failures.
                user_args = str(data.get("cli_args", "") or "")
                if self.check_diagnostics and self.diagnostic_flags:
                    run_args = (user_args + " " + " ".join(self.diagnostic_flags)).strip()
                else:
                    run_args = user_args

                try:
                    run_stdout = await mcp_client.run_executable(
                        executable=main_exec_name,
                        nsize=int(data.get("nsize", 1) or 1),
                        args=run_args,
                        timeout=self.smoke_run_timeout_sec,
                    )
                    # Clean exit — still check diagnostic output for silent failures.
                    if self.check_diagnostics:
                        diag = _parse_petsc_diagnostics(run_stdout or "", "")
                        if diag is not None:
                            last_failure_kind = f"diagnostic:{diag['kind']}"
                            last_failure_excerpt = diag["output_tail"]
                            print(
                                f"@@@ Purple agent v2: PETSc diagnostic flagged "
                                f"`{diag['kind']}` ({len(diag['matches'])} match(es))"
                            )
                            messages.append({
                                "role": "user",
                                "content": _format_diagnostic_failure(diag),
                            })
                            continue
                    data["self_verify_attempts"] = attempts_used
                    data["self_verify_status"] = "passed"
                    return data
                except petscmcp.MCPDynamicClientReturnCode as e:
                    # Non-zero exit from the binary itself. Try diagnostic parse first, fall back to raw stderr.
                    run_stderr = (e.stderr or e.stdout or "")[-3500:]
                    diag = (
                        _parse_petsc_diagnostics(e.stdout or "", e.stderr or "")
                        if self.check_diagnostics else None
                    )
                    if diag is not None:
                        last_failure_kind = f"diagnostic:{diag['kind']}"
                        last_failure_excerpt = diag["output_tail"]
                        print(
                            f"@@@ Purple agent v2: PETSc diagnostic on runtime failure "
                            f"(rc={e.returncode}, kind={diag['kind']})"
                        )
                        messages.append({
                            "role": "user",
                            "content": _format_diagnostic_failure(diag),
                        })
                    else:
                        last_failure_kind = "runtime"
                        last_failure_excerpt = run_stderr
                        hint = self._header_hint_block(run_stderr)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The code compiled but failed at runtime (return code {e.returncode}). "
                                f"stderr/stdout tail:\n```\n{run_stderr}\n```\n"
                                "Fix the runtime issue and return the full JSON object again."
                                f"{hint}"
                            ),
                        })
                    continue
                except petscmcp.MCPDynamicClientException as e:
                    # Genuine MCP infrastructure error (transport, server-side
                    # failure_message, ToolError). Cannot tell if the binary
                    # would have succeeded — don't ship potentially-broken
                    # code, but also don't loop forever on infra. Treat as
                    # runtime failure for this attempt and let the LLM try
                    # something different next iteration.
                    last_failure_kind = "runtime_mcp_infra"
                    last_failure_excerpt = str(e)
                    print(f"@@@ Purple agent v2: smoke run hit MCP infra error: {e}")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The smoke run failed due to a build-server infrastructure error: {e}. "
                            "This may indicate the binary did something unusual (timeout, abort, "
                            "or output the server could not handle). Return a corrected version "
                            "of the JSON object that runs to completion within a few seconds and "
                            "produces clean stdout."
                        ),
                    })
                    continue
        finally:
            # Best-effort MCP client teardown; ignore errors here so a failed close
            # doesn't mask the real failure that caused us to exit the loop.
            try:
                await mcp_ctx.__aexit__(None, None, None)
            except Exception:
                pass

        # Loop exhausted. Return the last attempt and let Green's outer retry
        # take another swing with its own feedback.
        elapsed = time.monotonic() - t0
        print(
            f"@@@ Purple agent v2: exhausted {attempts_used}/{iter_cap} self-fix attempts "
            f"in {elapsed:.0f}s (last failure: {last_failure_kind})"
        )
        # Only make the last-ditch LLM call if we have wall-clock room AND no
        # parseable data from any prior attempt. This closes a hang path where
        # the loop exits with last_data=None near Green's timeout and the extra
        # LLM call pushes the round-trip over the A2A read limit.
        if last_data is None:
            remaining = self.wall_clock_budget_sec - elapsed
            if remaining > 60.0:
                try:
                    last_data = self._call_llm(messages)
                except Exception as e:
                    print(f"@@@ Purple agent v2: last-ditch _call_llm failed ({e}); returning stub")
                    last_data = {"codes": [], "nsize": 1, "cli_args": ""}
            else:
                print(
                    f"@@@ Purple agent v2: skipping last-ditch _call_llm "
                    f"(only {remaining:.0f}s of budget left); returning stub to avoid A2A timeout"
                )
                last_data = {"codes": [], "nsize": 1, "cli_args": ""}
        last_data["self_verify_attempts"] = attempts_used
        last_data["self_verify_status"] = f"exhausted:{last_failure_kind}"
        return last_data

    async def _upload_files(self, mcp_client, codes: List[Dict[str, Any]], basename: str):
        """Upload generated files to the MCP server.

        The first file is renamed to {basename}.<ext> so the v2 agent's local
        executable name is predictable. Returns (main_exec_name, failure_msg_or_None).
        """
        if not codes:
            return (basename, "No code files in response.")
        try:
            for idx, entry in enumerate(codes):
                fname = entry.get("filename") or f"{basename}.c"
                # First file is the main file — rename to per-problem basename.
                if idx == 0:
                    ext = fname.rsplit('.', 1)[-1] if '.' in fname else 'c'
                    if ext == 'cu':
                        fname = f"{basename}cu.cu"
                    elif ext == 'cpp' and fname.endswith('kokkos.cpp'):
                        fname = f"{basename}kok.kokkos.cpp"
                    else:
                        fname = f"{basename}.c"
                code_str = entry.get("code", "")
                created = await mcp_client.create_file_from_string(
                    filename=fname, file_contents=code_str
                )
                if not created:
                    return (basename, f"MCP create_file_from_string returned false for {fname}")
            return (basename, None)
        except petscmcp.MCPDynamicClientException as e:
            return (basename, f"MCP error during file upload: {e}")

    async def cancel(self, context, event_queue) -> None:
        """Cancel a running task (not implemented).

        Required by the AgentExecutor interface but not implemented — v2's
        self-verify loop is atomic per problem and Green's outer timeout is
        the effective cancellation mechanism.

        Args:
            context: Request context
            event_queue: Event queue for sending cancellation acknowledgment

        Raises:
            NotImplementedError: Always, as cancellation is not supported.
        """
        raise NotImplementedError


def start_purple_agent_v2(
    host: str = "localhost",
    port: int = 9002,
    card_url: str | None = None,
    agent_llm: str | None = None,
    api_base_url: str | None = None,
    mcp_server_url: str | None = None,
    config_path: str = "config/purple_agent_v2_config.yaml",
):
    """Start the Purple Agent v2 A2A HTTP service.

    Args:
        host: Interface to bind the HTTP server to.
        port: Port to bind the HTTP server to.
        card_url: Optional explicit URL used to build/advertise the agent card.
            If not provided, a URL is constructed from `host` and `port`.
        agent_llm: Optional LLM model name (overrides config file value).
        api_base_url: Optional LLM API base URL (overrides config file value).
        mcp_server_url: Optional MCP compile-run server URL (overrides config file value).
        config_path: Path to the Purple Agent v2 config file (YAML/JSON).
    """
    logger.info("Starting purple agent v2...")
    card = prepare_purple_agent_v2_card(card_url or f"http://{host}:{port}")

    # Load config and apply CLI overrides.
    config = load_purple_agent_v2_config(config_path)
    if agent_llm:
        config.setdefault("llm", {})["model"] = agent_llm
    if api_base_url:
        config.setdefault("llm", {})["api_base_url"] = api_base_url
    if mcp_server_url:
        config.setdefault("mcp", {})["server_url"] = mcp_server_url

    request_handler = DefaultRequestHandler(
        agent_executor=PetscAgentV2Executor(config),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the purple agent v2 (self-verifying).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--config", type=str, default="config/purple_agent_v2_config.yaml")
    parser.add_argument("--api-base-url", type=str)
    parser.add_argument("--agent-llm", type=str)
    parser.add_argument("--mcp-server-url", type=str)
    args = parser.parse_args()

    start_purple_agent_v2(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        agent_llm=args.agent_llm,
        api_base_url=args.api_base_url,
        mcp_server_url=args.mcp_server_url,
        config_path=args.config,
    )
