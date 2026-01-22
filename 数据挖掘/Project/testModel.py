# ...existing code...
import joblib, pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

bundle = joblib.load('rdma_rf_model.pkl')
model, scaler = bundle['model'], bundle['scaler']

# 从 CSV 读取特征并随机抽取 10% 作为验证集
df = pd.read_csv('disaggregated_DLRM_trace.csv')
features = ['app_name', 'max_instance_per_node', 'memory_request', 'cpu_request']
label = 'rdma_request'  # 若目标列不同请修改
df = df[features + [label]].dropna()

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

X_val = val_df[features]
y_val = val_df[label].values

X_val_scaled = scaler.transform(X_val)
y_pred = model.predict(X_val_scaled)

# 输出结果与评估（同时打印分类准确率和回归 RMSE）
print('验证集样本数:', len(X_val))
# print('预测示例:', y_pred[:10])

# 如果适用，计算分类准确率（若为回归则忽略）
try:
    acc = accuracy_score(y_val, y_pred)
    print('Accuracy:', acc)
except Exception:
    pass

from sklearn.metrics import mean_absolute_error, r2_score

rmse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mean_y = np.mean(y_val) if len(y_val) else 0
rel_rmse = rmse / mean_y if mean_y != 0 else float('inf')

print('RMSE:', np.sqrt(rmse))
print('MAE:', mae)
print('R2:', r2)
print('相对RMSE (RMSE/mean):', rel_rmse)
# ...existing code...


# import joblib, pandas as pd
# bundle = joblib.load('rdma_rf_model.pkl')
# model, scaler = bundle['model'], bundle['scaler']

# new = pd.DataFrame({
#     'cpu_request': [16, 32],
#     'gpu_request': [1, 2],
#     'memory_request': [80.0, 240.0],
#     'disk_request': [80.0, 240.0]
# })
# new_scaled = scaler.transform(new)
# print('RDMA 预测:', model.predict(new_scaled))