import os, sys, json, subprocess, time
import numpy as np, pandas as pd, xgboost as xgb
from xgb_0_prep_params import features
from cv_folds import (CV_CITIES, TEST_CITIES, CITY_COUNTRIES, BUDGET, ES_ROUNDS,
                      get_folds, fold_id, load_cv_data, build_fold, build_all)

CV_MODE     = '6fold_st'                         # '12fold_loco' | '6fold_st'
MODELS_DIR  = f'{CV_MODE}_models'
URS_OUT     = f'urs_metrics_{CV_MODE}.csv'
INTERNAL_OUT= f'cv_internal_{CV_MODE}.csv'
FIT_FULL    = False                                  # also fit all-cities model, no eval set
FULL_OUT    = os.path.join(MODELS_DIR, f'xgb_{CV_MODE}_full.json')
os.makedirs(MODELS_DIR, exist_ok=True)

def run_script(script, desc, env):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    try:
        subprocess.run([sys.executable, script], check=True, env=env); return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {desc} (exit {e.returncode})"); return False

def _pool(g):
    n   = g['n'].sum()
    ssr = g['ss_res'].sum()
    sy, sy2 = g['sum_y'].sum(), g['sum_y2'].sum()
    ss_tot = sy2 - sy**2 / n                         # pooled SS_tot (global mean)
    return pd.Series({
        'rmse': np.sqrt(ssr / n),
        'r2':   1 - ssr / ss_tot,
        'bias': g['sum_res'].sum() / n,
        'n':    int(n),
    })

def summarize(path):
    df = pd.read_csv(path)
    print(f"\n{'#'*70}\nLOCO-CV SUMMARY\n{'#'*70}")

    # Per-city, per-fold (as measured, unweighted)
    print("\nPer-city RMSE  (rows=fold, cols=test_city):")
    print(df.pivot(index='fold', columns='test_city', values='rmse').round(3))
    print("\nPer-city R²:")
    print(df.pivot(index='fold', columns='test_city', values='r2').round(3))

    # Weighted per-fold: pool the 3 test cities by sample count
    per_fold = df.groupby('fold').apply(_pool)
    print("\nWeighted per-fold metrics (pooled over 3 test cities):")
    print(per_fold.round(3))

    # CV estimate + spread across folds
    print("\nCV estimate (mean ± std of weighted per-fold metrics, 12 folds):")
    for m in ['rmse', 'r2', 'bias']:
        print(f"  {m:>4}: {per_fold[m].mean():.3f} ± {per_fold[m].std():.3f}")


    print("\nPer test-city across folds:")
    print(df.groupby('test_city')[['rmse', 'r2', 'bias']]
            .agg(['mean', 'min', 'max']).round(3))

def main():
    params = json.load(open(f'best_hp_{CV_MODE}.json'))
    params.update({'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist'})
    data  = load_cv_data()
    folds = get_folds(CV_MODE)
    for f in (URS_OUT, INTERNAL_OUT):
        if os.path.exists(f): os.remove(f)

    internal = []
    for fold in folds:
        fid = fold_id(CV_MODE, fold)
        Xt, yt, Xv, yv = build_fold(CV_MODE, fold, data)
        print(f"\n### FOLD {fid}: train {Xt.shape[0]:,} rows | val {Xv.shape[0]:,} rows")
        if Xv.shape[0] == 0:
            print(f"  ⚠ EMPTY val fold {fid} — check coverage, skipping"); continue

        mp = os.path.join(MODELS_DIR, f'xgb_{CV_MODE}_{fid}.json')
        dtr = xgb.DMatrix(Xt, label=yt, feature_names=features)
        dvl = xgb.DMatrix(Xv, label=yv, feature_names=features)
        t0 = time.perf_counter()
        model = xgb.train(params, dtr, num_boost_round=BUDGET, evals=[(dvl,'valid')],
                          early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        train_s = time.perf_counter() - t0
        model.save_model(mp)
        internal.append({'fold':fid, 'cv_rmse':model.best_score,
                         'n_train':Xt.shape[0], 'n_val':Xv.shape[0],
                         'best_iter':model.best_iteration, 'train_s':round(train_s, 1)})
        print(f"  saved {mp} | internal CV RMSE {model.best_score:.4f} @ iter {model.best_iteration}"
              f" | trained in {train_s:.1f}s")

        # URS external test — reuse existing xgb_d_check unchanged
        for tc in TEST_CITIES:
            env = {**os.environ, 'TEST_TYPE':'urs', 'SPLIT_TYPE':'spatial',
                   'PIPELINE_MODEL_NAME':mp, 'PIPELINE_FOLD':fid,
                   'PIPELINE_METRICS_OUT':URS_OUT,
                   'PIPELINE_TARGET_CITY':tc, 'PIPELINE_TARGET_COUNTRY':CITY_COUNTRIES[tc]}
            run_script('xgb_d_check.py', f"{CV_MODE} {fid} → {tc}", env)

    pd.DataFrame(internal).to_csv(INTERNAL_OUT, index=False)
    print(f"\nInternal (spatio-temporal) CV metrics:\n{pd.DataFrame(internal).round(3)}")
    total_s = sum(r['train_s'] for r in internal)
    print(f"\nTotal training time ({len(internal)} folds): {total_s:.1f}s "
          f"({total_s/60:.1f} min) | mean/fold {total_s/len(internal):.1f}s")


    # ── Full model: fit on ALL training cities, no eval set ──────────────────
    if FIT_FULL and internal:
        n_round = int(np.median([r['best_iter'] for r in internal])) + 1   # trees = best_iter+1
        Xall, yall = build_all(data)
        print(f"\n### FULL FIT: {len(CV_CITIES)} cities | {Xall.shape[0]:,} rows | {n_round} rounds")
        full = xgb.train(params, xgb.DMatrix(Xall, label=yall, feature_names=features),
                         num_boost_round=n_round, verbose_eval=False)
        full.save_model(FULL_OUT)
        print(f"  saved {FULL_OUT}")


    summarize(URS_OUT)    

if __name__ == "__main__":
    main()