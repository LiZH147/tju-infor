# rdma_predictor.py   A B:1/2/3/4(迭代格式)/5（迭代格式讨论收敛性）
import pandas as pd
import numpy as np
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings, os
warnings.filterwarnings("ignore")

# 1. 载入数据
df = pd.read_csv('disaggregated_DLRM_trace.csv')

# 2. 清洗：去掉缺失、异常、无穷
cols = ['app_name', 'cpu_request','gpu_request', 'memory_request','rdma_request', 'disk_request','max_instance_per_node','creation_time','scheduled_time','deletion_time']
print('原始数据维度',df.shape)
df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
print('清洗后数据维度',df.shape)

# 3. 划分 X/y
# X = df[['app_name', 'disk_request','max_instance_per_node','gpu_request','memory_request','cpu_request']]
X = df[['app_name', 'max_instance_per_node', 'memory_request', 'cpu_request']]
y = df['rdma_request']

# 4. 标准化（树模型可不用，但后面 NN 会用，统一先 fit 好）
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 5. Optuna 自动调参
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    # 5 折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae = []
    for tr_idx, va_idx in kf.split(X_scaled):
        X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(X_tr, y_tr)
        mae.append(mean_absolute_error(y_va, model.predict(X_va)))
    return np.mean(mae)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print('Best MAE:', study.best_value)
print('Best params:', study.best_params)

# 6. 用最优参数重新训练
best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42)
best_model.fit(X_train, y_train)

# 7. 评估
pred = best_model.predict(X_test)
print('R2:', r2_score(y_test, pred))
print('MAE:', mean_absolute_error(y_test, pred))

# 8. SHAP 解释
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# 9. 保存模型 + 标准化器
joblib.dump({'model': best_model, 'scaler': scaler}, 'rdma_rf_model.pkl')
print('模型已保存为 rdma_rf_model.pkl')