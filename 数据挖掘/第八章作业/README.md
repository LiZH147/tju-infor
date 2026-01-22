
## 算法说明

### 1. K-Means (KMeans.py)
- **算法类型**: 基于划分的聚类算法
- **特点**: 简单高效，适用于球形簇
- **参数**: k（聚类数量）
- **数据文件**: `data/testSet.txt`
- **功能**: 
  - 随机初始化k个质心
  - 迭代更新质心和簇分配
  - 可视化聚类结果

### 2. PAM (PAM.py)
- **算法类型**: 基于划分的聚类算法（K-Medoids）
- **特点**: 使用实际数据点作为中心点，对异常值更鲁棒
- **参数**: k（聚类数量）
- **功能**:
  - 使用实际数据点作为中心点（medoids）
  - 通过交换中心点优化总成本
  - 可视化聚类结果和中心点

### 3. AGNES (AGNES.py)
- **算法类型**: 自底向上层次聚类算法
- **特点**: 逐步合并最相似的簇，生成层次结构
- **参数**: cluster（目标聚类数量）
- **数据文件**: `data/testSetAGNES.txt`
- **功能**:
  - 计算距离矩阵
  - 逐步合并最相似的簇
  - 绘制树状图（dendrogram）
  - 展示不同簇数下的聚类结果

### 4. DIANA (DIANA.py)
- **算法类型**: 自顶向下层次聚类算法
- **特点**: 从单个簇开始，逐步分裂成多个簇
- **参数**: k（目标聚类数量）
- **数据文件**: `data/testSet.txt`
- **功能**:
  - 找到具有最大直径的簇
  - 计算平均相异度
  - 将簇分裂为splinter group和old party

### 5. DBSCAN (DBSCAB.py)
- **算法类型**: 基于密度的聚类算法
- **特点**: 可以发现任意形状的簇，识别噪声点
- **参数**: 
  - Eps（邻域半径）
  - MinPts（最小点数）
- **数据文件**: `data/iris.data.csv`
- **功能**:
  - 基于密度进行聚类
  - 识别噪声点（NOISE）
  - 计算聚类评估指标（F-measure, ACC, NMI, RI, ARI）
  - 使用PCA降维可视化（当属性数>2时）

## 数据集说明

### iris.data.csv
- **描述**: 经典的鸢尾花数据集
- **特征**: 4个数值特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
- **类别**: 3类（Iris-setosa, Iris-versicolor, Iris-virginica）
- **用途**: 主要用于DBSCAN算法的测试和评估

### testSet.txt
- **描述**: 通用测试数据集
- **格式**: 空格分隔的数值数据
- **用途**: 用于K-Means、DIANA等算法的测试

### testSetAGNES.txt
- **描述**: AGNES算法专用测试数据集
- **格式**: 空格分隔的数值数据
- **用途**: 专门用于AGNES层次聚类算法的测试

## 使用方法

### 环境要求
```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## 参数调整

### DBSCAN参数
在 `DBSCAB.py` 中修改以下参数：
```python
Eps = 0.45      # 邻域半径
MinPts = 4      # 最小点数
```

### K-Means参数
在 `KMeans.py` 中修改：
```python
k_means(datMat, 4)  # 4为聚类数量
```

### AGNES参数
在 `AGNES.py` 中修改：
```python
model.train(cluster = 2)  # 2为聚类数量
```

### DIANA参数
在 `DIANA.py` 中修改：
```python
k = 2  # 目标聚类数量
```

## 输出说明

- **K-Means**: 显示质心坐标和散点图
- **PAM**: 显示迭代次数、总成本、聚类结果散点图
- **AGNES**: 显示不同簇数下的聚类结果和树状图
- **DIANA**: 显示分裂过程的详细信息
- **DBSCAN**: 显示聚类标签、评估指标（F-measure, ACC, NMI, RI, ARI）和可视化结果

## 算法比较

| 算法 | 簇形状 | 需要预设k | 处理噪声 | 时间复杂度 |
|------|--------|----------|----------|------------|
| K-Means | 球形 | 是 | 否 | O(nkt) |
| PAM | 球形 | 是 | 是 | O(k(n-k)²t) |
| AGNES | 任意 | 是 | 否 | O(n²log n) |
| DIANA | 任意 | 是 | 否 | O(n²log n) |
| DBSCAN | 任意 | 否 | 是 | O(n²) |

其中：n为数据点数，k为聚类数，t为迭代次数
