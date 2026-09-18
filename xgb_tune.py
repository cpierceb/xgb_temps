import json, numpy as np, optuna, xgboost as xgb
from xgb_0_prep_params import features
from cv_folds import (CV_CITIES, BUDGET, ES_ROUNDS,
                      get_folds, fold_id, load_cv_data, build_fold)

CV_MODE  = '12fold_loco'          # '12fold_loco' | '6fold_st'   ← the switch
N_TRIALS = 100
BEST_OUT = f'best_hp_{CV_MODE}.json'

data  = load_cv_data()                       # load 12 cities once
FOLDS = get_folds(CV_MODE)

def run_fold(params, fold):
    Xt, yt, Xv, yv = build_fold(CV_MODE, fold, data)
    dtr = xgb.DMatrix(Xt, label=yt, feature_names=features)
    dvl = xgb.DMatrix(Xv, label=yv, feature_names=features)
    model = xgb.train({**params, 'objective':'reg:squarederror', 'eval_metric':'rmse'},
                      dtr, num_boost_round=BUDGET, evals=[(dvl,'valid')],
                      early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    return model.best_score

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
        'tree_method':      'hist',
    }
    scores = []
    for i, fold in enumerate(FOLDS):
        scores.append(run_fold(params, fold))
        trial.report(float(np.mean(scores)), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


if __name__ == "__main__":
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest mean LOCO eval RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    with open(BEST_OUT, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved → {BEST_OUT}")