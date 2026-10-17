"""Walk-forward P(All-Star within 4 seasons) for not-yet-debuted prospects.

Same panel and learner as `light_model.py`. All-Star selections come from the
MLB Stats API cache (reference/mlb_awards.csv, prospects.data.sources.mlb_awards).

Label at landmark S: first All-Star selection in seasons S+1..S+4. A
prediction for season-end Y trains only on landmarks S <= Y-4, whose four
seasons are complete by Y. Writes raw/light_preds_as.csv.

    OMP_NUM_THREADS=2 python light_model_as.py
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from light_model import DB, HERE, fit_predict, load_panel

H = 4
PREDICT_YEARS = (2021, 2022, 2023)


def first_all_star() -> pd.Series:
    a = pd.read_csv(HERE.parent.parent / "reference" / "mlb_awards.csv")
    first = a[a.kind == "all_star"].groupby("mlbam_id").season.min()
    ids = pd.read_sql("select player_id, mlbam_id from prospects", sqlite3.connect(DB))
    ids["mlbam_id"] = pd.to_numeric(ids.mlbam_id, errors="coerce")
    return ids.set_index("player_id").mlbam_id.map(first)


def main():
    X, m = load_panel()
    m["as_first"] = m.player_id.map(first_all_star())
    S = m.S.to_numpy()
    y = ((m.as_first > m.S) & (m.as_first <= m.S + H)).to_numpy()
    out = []
    for Y in PREDICT_YEARS:
        train, test = S <= Y - H, S == Y
        d = m.loc[test, ["player_id", "S", "as_first"]].copy()
        d["p_as4"] = fit_predict(X, y, train, test)
        d["y_as4"] = y[test]
        seen = min(Y + H, 2026)  # outcomes observed through this season
        yy = d.y_as4.to_numpy()
        print(f"{Y}: train landmarks {int(train.sum()):,} ({int(y[train].sum())} positives) | "
              f"test n={len(d):,}, All-Stars by {seen}: {int(yy.sum())} | "
              f"AP {average_precision_score(yy, d.p_as4):.3f} AUC {roc_auc_score(yy, d.p_as4):.3f}"
              + ("" if Y + H <= 2026 else "  (window not complete)"))
        out.append(d)
    pd.concat(out).to_csv(HERE / "raw" / "light_preds_as.csv", index=False)


if __name__ == "__main__":
    main()
