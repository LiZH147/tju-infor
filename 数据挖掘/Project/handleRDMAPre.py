import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings('ignore')

# 1. 读数据、清洗、保留 4 列
df = pd.read_csv('disaggregated_DLRM_trace.csv')
cols = ['cpu_request','gpu_request','memory_request','rdma_request']
df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop('rdma_request', axis=1)
y = df['rdma_request']

# 2. 统一五折交叉验证
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import optuna

KF = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_mae(model, X, y):
    mae, r2, max_err = [], [], []
    for tr, va in KF.split(X):
        m = model.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        mae.append(mean_absolute_error(y.iloc[va], pred))
        r2.append(r2_score(y.iloc[va], pred))
        max_err.append(np.abs(y.iloc[va] - pred).max())
    return np.mean(mae), np.std(mae), np.mean(r2), np.std(max_err)

def rf_objective(trial):
    par = {
        'n_estimators': trial.suggest_int('n', 200, 1000),
        'max_depth': trial.suggest_int('d', 3, 20),
        'min_samples_split': trial.suggest_int('mss', 2, 20),
        'min_samples_leaf': trial.suggest_int('msl', 1, 10),
        'max_features': trial.suggest_categorical('mf', ['sqrt','log2', 0.5, 0.8]),
    }
    model = RandomForestRegressor(**par, random_state=42, n_jobs=-1)
    mae_mean, _, _, _ = cv_mae(model, X, y)
    return mae_mean

rf_study = optuna.create_study(direction='minimize')
rf_study.optimize(rf_objective, n_trials=30, show_progress_bar=True)

rf_best = RandomForestRegressor(**rf_study.best_params, random_state=42, n_jobs=-1)
rf_mae, rf_mae_std, rf_r2, rf_max = cv_mae(rf_best, X, y)
print(f'RF  MAE={rf_mae:.3f}±{rf_mae_std:.3f}  R2={rf_r2:.3f}  MaxError={rf_max:.3f}')
joblib.dump(rf_best, 'rf_rdema.pkl')

import lightgbm as lgb
def lgb_objective(trial):
    par = {
        'n_estimators': trial.suggest_int('n', 200, 1000),
        'num_leaves': trial.suggest_int('nl', 10, 100),
        'max_depth': trial.suggest_int('d', 3, 12),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('a', 0, 1),
        'reg_lambda': trial.suggest_float('l', 0, 1),
    }
    model = lgb.LGBMRegressor(**par, random_state=42, n_jobs=-1)
    mae_mean, _, _, _ = cv_mae(model, X, y)
    return mae_mean

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)

lgb_best = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1)
lgb_mae, lgb_mae_std, lgb_r2, lgb_max = cv_mae(lgb_best, X, y)
print(f'LGB MAE={lgb_mae:.3f}±{lgb_mae_std:.3f}  R2={lgb_r2:.3f}  MaxError={lgb_max:.3f}')
joblib.dump(lgb_best, 'lgb_rdema.pkl')

import xgboost as xgb
def xgb_objective(trial):
    par = {
        'n_estimators': trial.suggest_int('n', 200, 1000),
        'max_depth': trial.suggest_int('d', 3, 12),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('a', 0, 1),
        'reg_lambda': trial.suggest_float('l', 0, 1),
    }
    model = xgb.XGBRegressor(**par, random_state=42, n_jobs=-1)
    mae_mean, _, _, _ = cv_mae(model, X, y)
    return mae_mean

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)

xgb_best = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, n_jobs=-1)
xgb_mae, xgb_mae_std, xgb_r2, xgb_max = cv_mae(xgb_best, X, y)
print(f'XGB MAE={xgb_mae:.3f}±{xgb_mae_std:.3f}  R2={xgb_r2:.3f}  MaxError={xgb_max:.3f}')
joblib.dump(xgb_best, 'xgb_rdema.pkl')