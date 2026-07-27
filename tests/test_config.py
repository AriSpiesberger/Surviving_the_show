"""Tests for prospects.config — path, environment and run-namespace resolution.

Run:
    pytest tests/test_config.py
"""
from __future__ import annotations

import importlib

import pytest

from prospects import config


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep each test from inheriting a real RUN_TAG or a cached .env load."""
    monkeypatch.delenv("RUN_TAG", raising=False)
    monkeypatch.setattr(config, "_ENV_LOADED", True)  # skip real .env reads
    yield


# --- repo root -------------------------------------------------------------

def test_repo_root_is_the_directory_containing_the_package():
    assert (config.REPO_ROOT / "prospects" / "config.py").exists()


def test_repo_root_matches_regardless_of_import_path():
    # Guards the bug this module exists to prevent: modules at different
    # nesting depths computing different roots via parents[N].
    assert importlib.reload(config).REPO_ROOT == config.REPO_ROOT


# --- environment -----------------------------------------------------------

def test_load_env_does_not_override_existing_vars(tmp_path, monkeypatch):
    """setdefault semantics: run_job.ps1's exports must beat the .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("PROSPECT_DB=from_file\n", encoding="utf-8")
    monkeypatch.setenv("PROSPECT_DB", "from_environment")
    config.load_env(env_file, force=True)
    assert config.os.environ["PROSPECT_DB"] == "from_environment"


def test_load_env_sets_unset_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_UNSET_KEY=value\n", encoding="utf-8")
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    config.load_env(env_file, force=True)
    assert config.os.environ["SOME_UNSET_KEY"] == "value"


def test_load_env_ignores_comments_and_blanks(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nA_KEY=1\nnot_a_pair\n", encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    config.load_env(env_file, force=True)
    assert config.os.environ["A_KEY"] == "1"


def test_load_env_tolerates_missing_file(tmp_path):
    config.load_env(tmp_path / "does_not_exist", force=True)  # must not raise


def test_env_path_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("PROSPECT_DB", raising=False)
    assert config.live_db() == config.LIVE_DB


def test_env_path_honours_override(monkeypatch):
    monkeypatch.setenv("PROSPECT_DB", "/tmp/other.db")
    assert str(config.live_db()).endswith("other.db")


def test_live_and_model_db_are_distinct():
    assert config.LIVE_DB != config.MODEL_DB


# --- run namespaces --------------------------------------------------------

def test_run_paths_all_live_under_one_directory():
    p = config.RunPaths("v3")
    assert p.root == config.RUNS_DIR / "v3"
    for d in (p.models, p.training, p.scored, p.buy_lists, p.evaluation,
              p.scratch, p.logs):
        assert d.parent == p.root


def test_run_paths_are_tag_scoped():
    assert config.RunPaths("a").models != config.RunPaths("b").models


def test_current_tag_prefers_env(monkeypatch):
    monkeypatch.setenv("RUN_TAG", "from_env")
    assert config.current_tag() == "from_env"


def test_current_tag_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("RUN_TAG", raising=False)
    assert config.current_tag() == config.DEFAULT_TAG == "current"


def test_run_returns_explicit_tag_over_current(monkeypatch):
    monkeypatch.setenv("RUN_TAG", "current_one")
    assert config.run("explicit").tag == "explicit"
    assert config.run(None).tag == "current_one"


def test_mkdirs_creates_every_subdirectory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    p = config.RunPaths("t").mkdirs()
    for d in (p.models, p.training, p.scored, p.buy_lists, p.evaluation,
              p.scratch, p.logs):
        assert d.is_dir()


def test_canonical_artifact_names_carry_no_version_tag():
    p = config.RunPaths("current")
    for path in (p.hazards, p.hazards_landmark, p.joint_xgb, p.calibrators,
                 p.timing, p.lasso_logits, p.buy_list_final):
        # the point of the rename: no v1.18b / v2.0b / v3 in the filename
        assert not any(tok in path.name for tok in ("v1.", "v2.", "_v3")), path
    assert p.yip_thresholds(60).name == "yip_thresholds_p60.json"
    assert p.snap_long(2026).name == "snap2026_long.csv"


def test_artifacts_live_under_the_run_directory():
    p = config.RunPaths("current")
    assert p.joint_xgb.parent == p.models
    assert p.oof_val_long.parent == p.training
    assert p.buy_list_final.parent == p.buy_lists
