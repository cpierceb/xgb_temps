"""
Nested cross-validation for the 6-fold spatio-temporal CV.

"""
import os, json, time
import numpy as np, pandas as pd, optuna, xgboost as xgb
from xgb_0_prep_params import features
from cv_folds import (CV_CITIES, BUDGET, ES_ROUNDS,
                      ST_FOLDS, build_fold, load_cv_data)

CV_MODE          = '6fold_st'
N_TRIALS         = 100
MASK_OUTER_YEARS = True     # hide the outer period from inner-training too (recommended)
SEED             = 42
XGB_FIXED        = {'objective':'reg:squarederror', 'eval_metric':'rmse', 'tree_method':'hist'}

# Optional single-outer-fold mode for HPC parallelism (the 6 outer folds are
# independent). Set env NESTED_OUTER=st3 to run one fold; each job writes its own
# CSV, then concat the six and average outer_rmse. Unset -> run all six here.
OUTER_ONLY = os.environ.get('NESTED_OUTER')
OUTERS     = [f for f in ST_FOLDS if f['id'] == OUTER_ONLY] if OUTER_ONLY else list(ST_FOLDS)
OUT_CSV    = f'nested_cv_6fold_st{"_"+OUTER_ONLY if OUTER_ONLY else ""}.csv'
OUT_HP     = f'nested_hp_6fold_st{"_"+OUTER_ONLY if OUTER_ONLY else ""}.json'

data = load_cv_data()


def build_inner(outer, inner):
    """Inner train/val for one (outer, inner) pair, built from `data`.

    inner val   = inner-fold cities x inner-fold years            (exactly your block)
    inner train = the 8 cities outside BOTH folds, years outside the held period(s)

    The outer cities are excluded from inner-training always; with MASK_OUTER_YEARS
    the outer period is excluded too, so the outer holdout stays unseen in space AND
    time throughout tuning.
    """
    C_out, Y_out = set(outer['cities']), set(outer['years'])
    C_in,  Y_in  = set(inner['cities']), set(inner['years'])

    # inner VAL = held inner cities, restricted to their years
    Xv, yv = [], []
    for c in inner['cities']:
        X, y, yr = data[c]
        m = np.isin(yr, list(Y_in))
        Xv.append(X[m]); yv.append(y[m])
    Xv, yv = np.vstack(Xv), np.vstack(yv)

    # inner TRAIN = the other 8 cities, held periods masked
    excl = (Y_out | Y_in) if MASK_OUTER_YEARS else Y_in
    Xt, yt = [], []
    for c in CV_CITIES:
        if c in C_out or c in C_in:                 # both folds' cities out of training
            continue
        X, y, yr = data[c]
        m = ~np.isin(yr, list(excl))
        Xt.append(X[m]); yt.append(y[m])
    Xt, yt = np.vstack(Xt), np.vstack(yt)
    return Xt, yt, Xv, yv


def fit_inner(params, outer, inner):
    """One inner fit; early stopping on the inner-val set. Returns (rmse, best_iter)."""
    Xt, yt, Xv, yv = build_inner(outer, inner)
    dtr = xgb.DMatrix(Xt, label=yt, feature_names=features)
    dvl = xgb.DMatrix(Xv, label=yv, feature_names=features)
    m = xgb.train({**params, **XGB_FIXED}, dtr, num_boost_round=BUDGET,
                  evals=[(dvl, 'valid')], early_stopping_rounds=ES_ROUNDS,
                  verbose_eval=False)
    return m.best_score, m.best_iteration


def make_objective(outer, inner_folds):
    """Optuna objective = mean inner-val RMSE over the five inner folds."""
    def objective(trial):
        params = {
            'eta':              trial.suggest_float('eta', 0.05, 0.6, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        }
        scores = []
        for i, inner in enumerate(inner_folds):
            s, _ = fit_inner(params, outer, inner)
            scores.append(s)
            trial.report(float(np.mean(scores)), step=i)   # same pruning cadence as your tuner
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def main():
    rows, best_hps = [], {}

    for outer in OUTERS:
        oid = outer['id']
        inner_folds = [f for f in ST_FOLDS if f['id'] != oid]
        print(f"\n{'='*66}\nOUTER {oid}: hold {outer['cities']} x {outer['years']}"
              f"  | inner: {[f['id'] for f in inner_folds]}\n{'='*66}")

        # ---- inner tuning -----------------------------------------------------
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        t0 = time.perf_counter()
        study.optimize(make_objective(outer, inner_folds), n_trials=N_TRIALS)
        tune_s = time.perf_counter() - t0
        best = study.best_params
        best_hps[oid] = best
        print(f"  inner best mean RMSE {study.best_value:.4f}  |  tuned in {tune_s/60:.1f} min")

        # ---- refit rounds: re-run inner folds at best HP, take median best_iter
        #      (mirrors your FIT_FULL round-selection) --------------------------
        iters, inner_rmses = [], []
        for inner in inner_folds:
            s, bi = fit_inner(best, outer, inner)
            iters.append(bi); inner_rmses.append(s)
        n_round = int(np.median(iters)) + 1

        # ---- refit on the REAL outer-fold training set (10 cities, outer years
        #      masked = build_fold's Xt), rounds fixed, NO early stopping on outer
        Xt, yt, Xv, yv = build_fold(CV_MODE, outer, data)
        refit = xgb.train({**best, **XGB_FIXED},
                          xgb.DMatrix(Xt, label=yt, feature_names=features),
                          num_boost_round=n_round, verbose_eval=False)

        # ---- score the untouched outer holdout -------------------------------
        pred  = refit.predict(xgb.DMatrix(Xv, feature_names=features))
        resid = pred - yv.ravel()                    # == reconstructed-temp error (ERA5 cancels)
        rmse  = float(np.sqrt(np.mean(resid**2)))
        bias  = float(np.mean(resid))
        print(f"  n_round {n_round}  |  OUTER RMSE {rmse:.4f}  bias {bias:+.4f}"
              f"  |  n_train {Xt.shape[0]:,}  n_outer {Xv.shape[0]:,}")

        rows.append({
            'fold': oid,
            'cities': '+'.join(outer['cities']),
            'years': '-'.join(map(str, outer['years'])),
            'inner_mean_rmse': round(float(np.mean(inner_rmses)), 4),
            'n_round': n_round,
            'outer_rmse': round(rmse, 4),
            'outer_bias': round(bias, 4),
            'n_train': int(Xt.shape[0]),
            'n_outer': int(Xv.shape[0]),
            'tune_min': round(tune_s / 60, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    json.dump(best_hps, open(OUT_HP, 'w'), indent=2)

    print(f"\n{'#'*66}\nNESTED 6-FOLD CV{' ['+OUTER_ONLY+' only]' if OUTER_ONLY else ''}\n{'#'*66}")
    print(df.to_string(index=False))
    if not OUTER_ONLY:
        print(f"\nNested CV RMSE : {df.outer_rmse.mean():.3f} +/- {df.outer_rmse.std():.3f} K"
              f"   (n={len(df)} outer folds)")
        print(f"Nested CV bias : {df.outer_bias.mean():+.3f} +/- {df.outer_bias.std():.3f} K")
        print( "Compare against the tuned 6-fold CV RMSE reported in the paper (1.41 K).")
        print( "(R2 omitted on purpose: not invariant to the residual->temperature shift.)")
    else:
        print(f"\nSingle outer fold written to {OUT_CSV}. Concat the six and average "
              f"'outer_rmse' for the nested CV RMSE.")
    print(f"Saved -> {OUT_CSV}, {OUT_HP}")


if __name__ == "__main__":
    main()