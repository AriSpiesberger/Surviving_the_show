"""Repository paths, database resolution, environment, and run namespaces.

This module exists because configuration used to live in five uncoordinated
places: argparse defaults with hardcoded relative paths, module-level
``DB = "prospects_snapshot.db"`` constants, bare ``sqlite3.connect("...")``
calls, environment variables read only inside ``deploy/``, and ~20 copies of
``REPO_ROOT = Path(__file__).resolve().parents[N]`` with an inconsistent N.
That drift produced at least one real bug (a trainer loading the previous
generation's hazard file because two modules disagreed about its name).

Scope boundary: this module owns *where things live* — paths, the database,
the environment, and run namespaces. It does NOT own the model's feature
contract. ``prospects.model.joint`` owns EVENTS, FEAT_COND, H_MAX,
PUBLISH_H, AGE_CENTER and YIP_CENTER, and remains the single source of truth
for them so feature ordering can never drift. Import them from there, not
from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- repository root -------------------------------------------------------
# config.py sits at <repo>/prospects/config.py, so the root is one level up
# from the package. Every other module should import REPO_ROOT from here
# rather than recomputing parents[N].
REPO_ROOT = Path(__file__).resolve().parents[1]


# --- environment -----------------------------------------------------------
_ENV_LOADED = False


def load_env(path: Path | None = None, *, force: bool = False) -> None:
    """Load KEY=VALUE lines from .env into os.environ. Idempotent.

    Uses ``setdefault`` semantics: a variable already present in the
    environment WINS over the file. This is deliberate and load-bearing —
    ops/run_job.ps1 exports local path overrides (PROSPECT_DB, PRICES_DIR,
    ...) and those must not be clobbered by the checked-in .env defaults.
    Blank lines and ``#`` comments are ignored.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return
    env_path = path or (REPO_ROOT / ".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    _ENV_LOADED = True


def env_path(var: str, default: Path) -> Path:
    """Read a path-valued environment variable, falling back to `default`."""
    load_env()
    raw = os.environ.get(var)
    return Path(raw) if raw else default


# --- databases -------------------------------------------------------------
# Two SQLite files, with genuinely different roles:
#
#   prospects.db          the LIVE database. deploy/daily_data.py appends
#                         current-season stats to it every night.
#   prospects_snapshot.db the MODELING database. Despite the name it is not
#                         frozen: deploy/weekly_score.py copies the live DB
#                         over it at the top of each weekly run. It exists so
#                         a multi-hour training run reads a stable file while
#                         the nightly pull keeps writing to the live one.
#
# That weekly copy is the ONE supported bridge between them. Modeling code
# should read MODEL_DB; ingestion and the deploy jobs should use LIVE_DB.
LIVE_DB = REPO_ROOT / "prospects.db"
MODEL_DB = REPO_ROOT / "prospects_snapshot.db"


def live_db() -> Path:
    """The live database, overridable via the PROSPECT_DB env var."""
    return env_path("PROSPECT_DB", LIVE_DB)


def model_db() -> Path:
    """The modeling database, overridable via the PROSPECT_MODEL_DB env var."""
    return env_path("PROSPECT_MODEL_DB", MODEL_DB)


# --- top-level directories -------------------------------------------------
RUNS_DIR = REPO_ROOT / "runs"          # one subdirectory per model run
ARCHIVE_DIR = REPO_ROOT / "archive"    # superseded runs and retired artifacts
REFERENCE_DIR = REPO_ROOT / "reference"  # hand-curated static inputs
LOGS_DIR = REPO_ROOT / "logs"          # scheduled-job logs
PRICES_DIR_DEFAULT = REPO_ROOT / "prices"  # daily eBay pulls

# Pointer to the current run. A plain text file holding a tag name rather
# than a symlink/junction: it needs no special privileges, survives archive
# operations, is greppable, and can be committed.
CURRENT_TAG_FILE = RUNS_DIR / "CURRENT"
DEFAULT_TAG = "prod"


# --- buy-list policy defaults ---------------------------------------------
# Policy knobs (what we choose to publish), as distinct from the model's
# feature contract in joint_cond. DEFAULT_THRESHOLD was previously hardcoded
# as a bare 0.60 in seven places across the buy-list builder and evaluators.
DEFAULT_THRESHOLD = 0.60
DEFAULT_DEBUT_HORIZON = 3


# --- run namespaces --------------------------------------------------------
@dataclass(frozen=True)
class RunPaths:
    """Every artifact directory belonging to one model run.

    A "run" is one pass of the pipeline under a tag (``prod``, ``v3``,
    ``partial``, ...). Previously a single run scattered its outputs across
    models/, results/scored/, results/buy_lists/, results/training/,
    evaluation/<tag>/ and scratch/v20b_oof_<tag>/, and the tag was applied by
    mutating module globals in five files independently. Now one run is one
    directory:

        runs/<tag>/
          models/      trained artifacts (hazards, joint XGB, calibrators)
          training/    fit/val longs, panel caches, pid lists
          scored/      snap scoring output
          buy_lists/   the published lists
          evaluation/  metric tables + generated README
          scratch/     regeneratable intermediates
          logs/        per-run training logs

    Because the directory carries the tag, the filenames inside it do not
    need to — `runs/v3/models/joint_xgb_oof.pkl`, not
    `models/joint_xgb_v2.0b_v3_oof.pkl`.
    """

    tag: str

    @property
    def root(self) -> Path:
        return RUNS_DIR / self.tag

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def scored(self) -> Path:
        return self.root / "scored"

    @property
    def buy_lists(self) -> Path:
        return self.root / "buy_lists"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def mkdirs(self) -> RunPaths:
        """Create every subdirectory. Returns self so it can be chained."""
        for d in (self.models, self.training, self.scored, self.buy_lists,
                  self.evaluation, self.scratch, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def current(cls) -> RunPaths:
        """The run the pipeline reads and writes by default.

        Resolution order: the RUN_TAG environment variable, then the
        runs/CURRENT pointer file, then DEFAULT_TAG.
        """
        return cls(current_tag())


def current_tag() -> str:
    """Tag of the current run (env RUN_TAG > runs/CURRENT > DEFAULT_TAG)."""
    load_env()
    tag = os.environ.get("RUN_TAG")
    if tag:
        return tag.strip()
    if CURRENT_TAG_FILE.exists():
        tag = CURRENT_TAG_FILE.read_text(encoding="utf-8").strip()
        if tag:
            return tag
    return DEFAULT_TAG


def set_current_tag(tag: str) -> None:
    """Point runs/CURRENT at `tag`, creating the run directory if needed."""
    RunPaths(tag).mkdirs()
    CURRENT_TAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_TAG_FILE.write_text(tag.strip() + "\n", encoding="utf-8")


def run(tag: str | None = None) -> RunPaths:
    """RunPaths for `tag`, or for the current run when tag is None.

    The standard way for a script to accept a --tag flag:

        ap.add_argument("--tag", default=None)
        paths = config.run(args.tag)
    """
    return RunPaths(tag) if tag else RunPaths.current()
