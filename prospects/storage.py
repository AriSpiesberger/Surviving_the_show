"""
prospects/storage.py
======================

SQLite layer for the prospect database.

Tables:
    prospects           - one row per player, current snapshot
    season_stats        - one row per player-season-level
    career_outcomes     - one row per resolved player (training labels)
    rankings_history    - one row per player-source-date
    predictions         - one row per classifier run per player
    event_multipliers   - card price multipliers per event (size model)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from prospects.schema import (
    CareerEvent,
    CareerOutcome,
    EventMultiplier,
    EventProbability,
    Prospect,
    ProspectPrediction,
    RankingSnapshot,
    SeasonStats,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prospects (
    player_id TEXT PRIMARY KEY,
    mlbam_id TEXT,
    name TEXT NOT NULL,
    is_pitcher INTEGER NOT NULL,
    primary_position TEXT NOT NULL,
    birth_date TEXT,

    -- Pedigree
    draft_year INTEGER,
    draft_round INTEGER,
    draft_pick INTEGER,
    signing_bonus_usd REAL,
    age_at_signing REAL,
    is_international INTEGER DEFAULT 0,
    international_signing_year INTEGER,
    origin TEXT,

    -- Current state
    current_org TEXT,
    current_level TEXT,
    highest_level_reached TEXT,

    -- Risk
    tj_history INTEGER DEFAULT 0,
    has_current_injury INTEGER DEFAULT 0,
    current_injury_type TEXT,

    -- Metadata
    notes TEXT,
    as_of_date TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prospects_org ON prospects(current_org);
CREATE INDEX IF NOT EXISTS idx_prospects_pos ON prospects(primary_position);
CREATE INDEX IF NOT EXISTS idx_prospects_pitcher ON prospects(is_pitcher);

CREATE TABLE IF NOT EXISTS season_stats (
    player_id TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    level TEXT NOT NULL,
    org TEXT,
    age_during_season REAL,

    -- Hitter
    pa INTEGER DEFAULT 0,
    avg REAL, obp REAL, slg REAL, woba REAL, iso REAL,
    k_pct REAL, bb_pct REAL, babip REAL,
    home_runs INTEGER, stolen_bases INTEGER,

    -- Pitcher
    ip REAL DEFAULT 0,
    era REAL, fip REAL, whip REAL,
    k9 REAL, bb9 REAL, hr9 REAL, velo_avg REAL,

    primary_position TEXT,
    PRIMARY KEY (player_id, season_year, level)
);

CREATE INDEX IF NOT EXISTS idx_season_year ON season_stats(season_year);
CREATE INDEX IF NOT EXISTS idx_season_level ON season_stats(level);

CREATE TABLE IF NOT EXISTS career_outcomes (
    player_id TEXT PRIMARY KEY,
    career_complete INTEGER NOT NULL,
    career_pa INTEGER DEFAULT 0,
    career_ip REAL DEFAULT 0,
    career_war REAL DEFAULT 0,
    all_star_selections INTEGER DEFAULT 0,
    mvp_count INTEGER DEFAULT 0,
    cy_young_count INTEGER DEFAULT 0,
    roy_count INTEGER DEFAULT 0,
    is_hof_inducted INTEGER DEFAULT 0,
    is_hof_likely INTEGER DEFAULT 0,
    best_overall_rank INTEGER,
    pro_debut_year INTEGER,
    mlb_debut_year INTEGER,
    final_mlb_year INTEGER,
    -- First-trigger years per event (NULL if never triggered).
    -- Used to mask "already-triggered-at-as-of" samples during training.
    year_top_100 INTEGER,
    year_top_25 INTEGER,
    year_established_mlb INTEGER,
    year_all_star_once INTEGER,
    year_all_star_three INTEGER,
    year_major_award INTEGER,
    year_hof_trajectory INTEGER,
    events_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rankings_history (
    player_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    source TEXT NOT NULL,
    overall_rank INTEGER,
    org_rank INTEGER,
    list_size INTEGER DEFAULT 100,
    PRIMARY KEY (player_id, as_of, source)
);

CREATE INDEX IF NOT EXISTS idx_rankings_source ON rankings_history(source, as_of);

CREATE TABLE IF NOT EXISTS predictions (
    player_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    model_version TEXT NOT NULL,
    events_json TEXT NOT NULL,
    confidence REAL,
    features_used INTEGER,
    features_imputed INTEGER,
    PRIMARY KEY (player_id, as_of_date, model_version)
);

CREATE TABLE IF NOT EXISTS event_multipliers (
    event_id INTEGER NOT NULL,
    product_family TEXT NOT NULL,
    parallel_tier TEXT NOT NULL,
    multiplier_mean REAL NOT NULL,
    multiplier_stdev REAL NOT NULL,
    n_observations INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (event_id, product_family, parallel_tier)
);
"""


class ProspectDB:
    """SQLite-backed storage."""

    def __init__(self, db_path: str = "prospects.db"):
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # Lightweight forward migrations for existing DBs.
            cols = {r["name"] for r in conn.execute(
                "PRAGMA table_info(prospects)").fetchall()}
            if "mlbam_id" not in cols:
                conn.execute("ALTER TABLE prospects ADD COLUMN mlbam_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prospects_mlbam ON prospects(mlbam_id)"
            )
            ocols = {r["name"] for r in conn.execute(
                "PRAGMA table_info(career_outcomes)").fetchall()}
            for col in (
                "year_top_100", "year_top_25", "year_established_mlb",
                "year_all_star_once", "year_all_star_three",
                "year_major_award", "year_hof_trajectory",
            ):
                if col not in ocols:
                    conn.execute(f"ALTER TABLE career_outcomes ADD COLUMN {col} INTEGER")
            scols = {r["name"] for r in conn.execute(
                "PRAGMA table_info(season_stats)").fetchall()}
            if "games_played" not in scols:
                conn.execute("ALTER TABLE season_stats ADD COLUMN games_played INTEGER")
            if "season_complete" not in scols:
                conn.execute("ALTER TABLE season_stats ADD COLUMN season_complete INTEGER")
            if "injury_suspected" not in scols:
                conn.execute("ALTER TABLE season_stats ADD COLUMN injury_suspected INTEGER")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ========================================================================
    # PROSPECTS
    # ========================================================================

    def upsert_prospect(self, p: Prospect) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prospects (
                    player_id, name, is_pitcher, primary_position, birth_date,
                    draft_year, draft_round, draft_pick, signing_bonus_usd,
                    age_at_signing, is_international, international_signing_year,
                    origin, current_org, current_level, highest_level_reached,
                    tj_history, has_current_injury, current_injury_type,
                    notes, as_of_date, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    name=excluded.name,
                    is_pitcher=excluded.is_pitcher,
                    primary_position=excluded.primary_position,
                    birth_date=excluded.birth_date,
                    draft_year=excluded.draft_year,
                    draft_round=excluded.draft_round,
                    draft_pick=excluded.draft_pick,
                    signing_bonus_usd=excluded.signing_bonus_usd,
                    age_at_signing=excluded.age_at_signing,
                    is_international=excluded.is_international,
                    international_signing_year=excluded.international_signing_year,
                    origin=excluded.origin,
                    current_org=excluded.current_org,
                    current_level=excluded.current_level,
                    highest_level_reached=excluded.highest_level_reached,
                    tj_history=excluded.tj_history,
                    has_current_injury=excluded.has_current_injury,
                    current_injury_type=excluded.current_injury_type,
                    notes=excluded.notes,
                    as_of_date=excluded.as_of_date,
                    updated_at=excluded.updated_at
                """,
                (
                    p.player_id, p.name, int(p.is_pitcher), p.primary_position,
                    p.birth_date.isoformat() if p.birth_date else None,
                    p.pedigree.draft_year, p.pedigree.draft_round, p.pedigree.draft_pick,
                    p.pedigree.signing_bonus_usd, p.pedigree.age_at_signing,
                    int(p.pedigree.is_international), p.pedigree.international_signing_year,
                    p.pedigree.origin,
                    p.current_org, p.current_level, p.highest_level_reached,
                    int(p.risk.tj_history), int(p.risk.has_current_injury),
                    p.risk.current_injury_type,
                    p.notes,
                    p.as_of_date.isoformat() if p.as_of_date else None,
                    datetime.utcnow().isoformat(),
                ),
            )

        # Rankings stored separately
        for ranking in p.rankings:
            self.upsert_ranking(p.player_id, ranking)

    def get_prospect(self, player_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prospects WHERE player_id = ?", (player_id,)
            ).fetchone()
        return dict(row) if row else None

    def all_prospects(self, org: Optional[str] = None) -> list[dict]:
        with self._connect() as conn:
            if org:
                rows = conn.execute(
                    "SELECT * FROM prospects WHERE current_org = ? ORDER BY name",
                    (org,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM prospects ORDER BY name"
                ).fetchall()
        return [dict(r) for r in rows]

    def count_prospects(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM prospects").fetchone()[0]

    def set_mlbam_id(self, player_id: str, mlbam_id: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE prospects SET mlbam_id = ? WHERE player_id = ?",
                (mlbam_id, player_id),
            )

    def prospect_by_mlbam(self, mlbam_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prospects WHERE mlbam_id = ?", (mlbam_id,)
            ).fetchone()
        return dict(row) if row else None

    # ========================================================================
    # SEASON STATS
    # ========================================================================

    def upsert_season_stats(self, s: SeasonStats) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO season_stats (
                    player_id, season_year, level, org, age_during_season,
                    pa, avg, obp, slg, woba, iso,
                    k_pct, bb_pct, babip, home_runs, stolen_bases,
                    ip, era, fip, whip,
                    k9, bb9, hr9, velo_avg,
                    primary_position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, season_year, level) DO UPDATE SET
                    org=excluded.org,
                    age_during_season=excluded.age_during_season,
                    pa=excluded.pa,
                    avg=excluded.avg, obp=excluded.obp, slg=excluded.slg,
                    woba=excluded.woba, iso=excluded.iso,
                    k_pct=excluded.k_pct, bb_pct=excluded.bb_pct, babip=excluded.babip,
                    home_runs=excluded.home_runs, stolen_bases=excluded.stolen_bases,
                    ip=excluded.ip, era=excluded.era, fip=excluded.fip,
                    whip=excluded.whip,
                    k9=excluded.k9, bb9=excluded.bb9, hr9=excluded.hr9,
                    velo_avg=excluded.velo_avg,
                    primary_position=excluded.primary_position
                """,
                (
                    s.player_id, s.season_year, s.level, s.org, s.age_during_season,
                    s.pa, s.avg, s.obp, s.slg, s.woba, s.iso,
                    s.k_pct, s.bb_pct, s.babip, s.home_runs, s.stolen_bases,
                    s.ip, s.era, s.fip, s.whip,
                    s.k9, s.bb9, s.hr9, s.velo_avg,
                    s.primary_position,
                ),
            )

    def get_season_stats(self, player_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM season_stats WHERE player_id = ? ORDER BY season_year, level",
                (player_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_season_stats(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM season_stats").fetchone()[0]

    # ========================================================================
    # RANKINGS
    # ========================================================================

    def upsert_ranking(self, player_id: str, r: RankingSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rankings_history (
                    player_id, as_of, source, overall_rank, org_rank, list_size
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, as_of, source) DO UPDATE SET
                    overall_rank=excluded.overall_rank,
                    org_rank=excluded.org_rank,
                    list_size=excluded.list_size
                """,
                (
                    player_id, r.as_of.isoformat(), r.source,
                    r.overall_rank, r.org_rank, r.list_size,
                ),
            )

    def get_rankings(self, player_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rankings_history WHERE player_id = ? ORDER BY as_of",
                (player_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def best_rank(self, player_id: str) -> Optional[int]:
        """Returns the best (lowest number) overall rank ever achieved."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(overall_rank) as best
                FROM rankings_history
                WHERE player_id = ? AND overall_rank IS NOT NULL
                """,
                (player_id,),
            ).fetchone()
        return row["best"] if row and row["best"] is not None else None

    # ========================================================================
    # CAREER OUTCOMES
    # ========================================================================

    def upsert_outcome(self, o: CareerOutcome) -> None:
        events_json = json.dumps(
            {str(int(e)): triggered for e, triggered in o.events.items()}
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO career_outcomes (
                    player_id, career_complete,
                    career_pa, career_ip, career_war,
                    all_star_selections, mvp_count, cy_young_count, roy_count,
                    is_hof_inducted, is_hof_likely, best_overall_rank,
                    pro_debut_year, mlb_debut_year, final_mlb_year,
                    events_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    career_complete=excluded.career_complete,
                    career_pa=excluded.career_pa,
                    career_ip=excluded.career_ip,
                    career_war=excluded.career_war,
                    all_star_selections=excluded.all_star_selections,
                    mvp_count=excluded.mvp_count,
                    cy_young_count=excluded.cy_young_count,
                    roy_count=excluded.roy_count,
                    is_hof_inducted=excluded.is_hof_inducted,
                    is_hof_likely=excluded.is_hof_likely,
                    best_overall_rank=excluded.best_overall_rank,
                    pro_debut_year=excluded.pro_debut_year,
                    mlb_debut_year=excluded.mlb_debut_year,
                    final_mlb_year=excluded.final_mlb_year,
                    events_json=excluded.events_json,
                    updated_at=excluded.updated_at
                """,
                (
                    o.player_id, int(o.career_complete),
                    o.career_pa, o.career_ip, o.career_war,
                    o.all_star_selections, o.mvp_count, o.cy_young_count, o.roy_count,
                    int(o.is_hof_inducted), int(o.is_hof_likely), o.best_overall_rank,
                    o.pro_debut_year, o.mlb_debut_year, o.final_mlb_year,
                    events_json,
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_outcome(self, player_id: str) -> Optional[CareerOutcome]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM career_outcomes WHERE player_id = ?", (player_id,)
            ).fetchone()
        if not row:
            return None
        events_raw = json.loads(row["events_json"]) if row["events_json"] else {}
        events = {CareerEvent(int(k)): bool(v) for k, v in events_raw.items()}
        return CareerOutcome(
            player_id=row["player_id"],
            career_complete=bool(row["career_complete"]),
            career_pa=row["career_pa"],
            career_ip=row["career_ip"],
            career_war=row["career_war"],
            all_star_selections=row["all_star_selections"],
            mvp_count=row["mvp_count"],
            cy_young_count=row["cy_young_count"],
            roy_count=row["roy_count"],
            is_hof_inducted=bool(row["is_hof_inducted"]),
            is_hof_likely=bool(row["is_hof_likely"]),
            best_overall_rank=row["best_overall_rank"],
            pro_debut_year=row["pro_debut_year"],
            mlb_debut_year=row["mlb_debut_year"],
            final_mlb_year=row["final_mlb_year"],
            events=events,
        )

    def all_outcomes(self) -> list[CareerOutcome]:
        with self._connect() as conn:
            ids = [r["player_id"] for r in conn.execute(
                "SELECT player_id FROM career_outcomes"
            ).fetchall()]
        return [self.get_outcome(pid) for pid in ids]

    def count_outcomes(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM career_outcomes").fetchone()[0]

    def set_event_trigger_years(self, player_id: str, years: dict) -> None:
        """Update per-event trigger years for one player. `years` keys:
        year_top_100, year_top_25, year_established_mlb, year_all_star_once,
        year_all_star_three, year_major_award, year_hof_trajectory."""
        cols = [
            "year_top_100", "year_top_25", "year_established_mlb",
            "year_all_star_once", "year_all_star_three",
            "year_major_award", "year_hof_trajectory",
        ]
        sets = ", ".join(f"{c} = ?" for c in cols)
        params = [years.get(c) for c in cols] + [player_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE career_outcomes SET {sets} WHERE player_id = ?",
                params,
            )

    # ========================================================================
    # PREDICTIONS
    # ========================================================================

    def insert_prediction(self, p: ProspectPrediction) -> None:
        events_json = json.dumps({
            str(int(e)): {
                "mean": ep.p_mean, "lo": ep.p_lo, "hi": ep.p_hi
            }
            for e, ep in p.events.items()
        })
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    player_id, as_of_date, model_version, events_json,
                    confidence, features_used, features_imputed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, as_of_date, model_version) DO UPDATE SET
                    events_json=excluded.events_json,
                    confidence=excluded.confidence,
                    features_used=excluded.features_used,
                    features_imputed=excluded.features_imputed
                """,
                (
                    p.player_id, p.as_of_date.isoformat(),
                    p.model_version, events_json,
                    p.confidence, p.features_used, p.features_imputed,
                ),
            )

    def get_latest_prediction(
        self, player_id: str, model_version: Optional[str] = None
    ) -> Optional[ProspectPrediction]:
        q = "SELECT * FROM predictions WHERE player_id = ?"
        params: list = [player_id]
        if model_version:
            q += " AND model_version = ?"
            params.append(model_version)
        q += " ORDER BY as_of_date DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(q, params).fetchone()
        if not row:
            return None
        events_raw = json.loads(row["events_json"])
        events = {
            CareerEvent(int(k)): EventProbability(
                event=CareerEvent(int(k)),
                p_mean=v["mean"], p_lo=v["lo"], p_hi=v["hi"],
            )
            for k, v in events_raw.items()
        }
        return ProspectPrediction(
            player_id=row["player_id"],
            as_of_date=date.fromisoformat(row["as_of_date"]),
            events=events,
            confidence=row["confidence"],
            model_version=row["model_version"],
            features_used=row["features_used"],
            features_imputed=row["features_imputed"],
        )

    # ========================================================================
    # EVENT MULTIPLIERS (size model)
    # ========================================================================

    def upsert_multiplier(
        self, m: EventMultiplier, product_family: str, parallel_tier: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO event_multipliers (
                    event_id, product_family, parallel_tier,
                    multiplier_mean, multiplier_stdev, n_observations, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, product_family, parallel_tier) DO UPDATE SET
                    multiplier_mean=excluded.multiplier_mean,
                    multiplier_stdev=excluded.multiplier_stdev,
                    n_observations=excluded.n_observations,
                    updated_at=excluded.updated_at
                """,
                (
                    int(m.event), product_family, parallel_tier,
                    m.multiplier_mean, m.multiplier_stdev, m.n_observations,
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_multipliers(
        self, product_family: str, parallel_tier: str
    ) -> dict[CareerEvent, EventMultiplier]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM event_multipliers
                WHERE product_family = ? AND parallel_tier = ?
                """,
                (product_family, parallel_tier),
            ).fetchall()
        return {
            CareerEvent(r["event_id"]): EventMultiplier(
                event=CareerEvent(r["event_id"]),
                multiplier_mean=r["multiplier_mean"],
                multiplier_stdev=r["multiplier_stdev"],
                n_observations=r["n_observations"],
            )
            for r in rows
        }
