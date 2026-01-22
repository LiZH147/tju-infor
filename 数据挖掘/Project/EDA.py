# # --------------------------------------------------
# # 1. 基础导入 & 数据读取
# # --------------------------------------------------
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-darkgrid")
# sns.set_context("talk")   # 图字体更大，适合汇报

# df = pd.read_csv("disaggregated_DLRM_trace.csv")

# # 只保留我们关心的 4 列
# COLS = ["cpu_request", "gpu_request", "memory_request", "rdma_request"]
# eda = df[COLS].copy()

# # 基础清洗：去 inf/NA
# eda = eda.replace([np.inf, -np.inf], np.nan).dropna()
# eda.head()

# # --------------------------------------------------
# # 2. 一维分布：rdma_request 三峰明显
# # --------------------------------------------------
# plt.figure(figsize=(8, 5))
# sns.histplot(eda["rdma_request"], bins=30, kde=True, color="#1f77b4")
# plt.title("RDMA Request Distribution")
# plt.xlabel("rdma_request (Gbps)")
# plt.ylabel("Count")
# plt.yscale("log")
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------
# # 3. 二维关系：gpu_request 是决定性因子
# # --------------------------------------------------
# plt.figure(figsize=(6, 5))
# sns.boxplot(x="gpu_request", y="rdma_request", data=eda, palette="Set2")
# plt.title("RDMA vs GPU request")
# plt.ylabel("rdma_request (Gbps)")
# plt.show()

# # --------------------------------------------------
# # 5. 三维散点：cpu / memory 的微调作用
# # --------------------------------------------------
# from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(111, projection="3d")
# # 用 gpu 做颜色映射
# scatter = ax.scatter(
#     eda["cpu_request"],
#     eda["memory_request"],
#     eda["rdma_request"],
#     c=eda["gpu_request"],
#     cmap="viridis",
#     alpha=0.7,
#     s=20,
# )
# ax.set_xlabel("CPU request")
# ax.set_ylabel("Memory request (GiB)")
# ax.set_zlabel("RDMA request (Gbps)")
# plt.colorbar(scatter, label="gpu_request")
# plt.title("3D Scatter: CPU / Memory / RDMA (colored by GPU)")
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------
# # 4. 相关性热力图
# # --------------------------------------------------
# plt.figure(figsize=(5, 4))
# corr = eda.corr()
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
# plt.title("Pearson Correlation Matrix")
# plt.show()

import pandas as pd, numpy as np, warnings
# warnings.filterfilter('ignore')

df = pd.read_csv('disaggregated_DLRM_trace.csv')

# 把你能拿到的所有“请求/限制”字段都放进来候选
cand = ['app_name','cpu_request','gpu_request',
        'memory_request','rdma_request', 'disk_request','max_instance_per_node','creation_time','scheduled_time','deletion_time']
cand = [c for c in cand if c in df.columns]
Xy = df[cand].dropna()
X = Xy.drop('rdma_request', axis=1)
y = Xy['rdma_request']

import seaborn as sns, matplotlib.pyplot as plt

corr_lin  = X.corrwith(y)               # Pearson
corr_mon  = X.corrwith(y, method='spearman')  # Spearman

plt.figure(figsize=(4,3))
sns.barplot(x=corr_lin.abs().sort_values(ascending=False),
            y=corr_lin.abs().sort_values(ascending=False).index,
            color='steelblue')
plt.title('|Pearson| with rdma_request')
plt.show()

from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y, random_state=42)
mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi)

import shap, optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# 用 Optuna 快速调一个轻量 RF
def obj(trial):
    par = {'n_estimators':trial.suggest_int('n',100,800),
           'max_depth':trial.suggest_int('d',3,15),
           'min_samples_leaf':trial.suggest_int('msl',1,10)}
    m = RandomForestRegressor(**par, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    return -m.score(X_test, y_test)   # 负 R2

study = optuna.create_study()
study.optimize(obj, n_trials=30, show_progress_bar=False)

best = RandomForestRegressor(**study.best_params, random_state=42).fit(X_train, y_train)
explainer = shap.TreeExplainer(best)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
