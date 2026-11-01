import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, debut_pop as dp
from panel import player_panel
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, RidgeCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
d = dp.debuts('raw'); d = d[d.p_pre >= 3]
x = dp.features(d)
m = player_panel(grade='raw'); st = m.set_index(['player_id', 'month']).stale
x['fresh'] = [not st.get((p, mo - 1), True) for p, mo in zip(x.player_id, x.debut_month)]
x = x.reset_index(drop=True)
y = x['pop'].clip(-1.5, 2.5).to_numpy()
X = dp.design(x)
noyr = [c for c in X if not c.startswith('debut_year_')]
noteam = [c for c in noyr if not c.startswith('team_')]
core = ['log_p_pre', 'top25', 'ever_top100', 'log_best_rank', 'sept', 'april', 'card_age', 'log_psa_ratio', 'age', 'r1', 'ifa', 'skipped_aaa', 'p_deb1'] + [c for c in X if c.startswith('pos_')]
def run(name, cols, model, train_mask=None, eval_mask=None):
    pred = np.full(len(x), np.nan)
    for yr in sorted(x.debut_year.unique()):
        te = (x.debut_year == yr).to_numpy(); tr = ~te
        if train_mask is not None: tr = tr & train_mask
        if te.sum() < 10: continue
        mdl = make_pipeline(StandardScaler(), model()).fit(X.loc[tr, cols], y[tr])
        pred[te] = mdl.predict(X.loc[te, cols])
    ok = ~np.isnan(pred) & (eval_mask if eval_mask is not None else True)
    yy, pp = y[ok], pred[ok]
    r2 = 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    q = pd.qcut(pp, 5, labels=False)
    top, bot = yy[q == 4].mean(), yy[q == 0].mean()
    print(f"{name:42s} n={ok.sum():3d} R2={r2:+.3f} rho={spearmanr(pp, yy)[0]:+.3f}  top-quintile {top:+.3f} bottom {bot:+.3f}")
    return pred
fr = x.fresh.to_numpy()
L = lambda: LassoCV(cv=5, n_alphas=40, max_iter=20000, random_state=0)
R = lambda: RidgeCV(alphas=np.logspace(-1, 4, 30))
for nm, cols in (('all feats', noyr), ('no team', noteam), ('core', core)):
    run(f'lasso {nm}', cols, L)
    run(f'ridge {nm}', cols, R)
    run(f'ridge {nm} fresh-train fresh-eval', cols, R, fr, fr)
    run(f'lasso {nm} fresh-train fresh-eval', cols, L, fr, fr)
