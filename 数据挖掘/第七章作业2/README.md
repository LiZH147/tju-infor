# SVM / KNN / BP

## 简介
- 包含三个简单的机器学习实现：支持向量机 (SVM)、K 最近邻 (KNN) 和一个三层 BP 神经网络 (BP)。
- 示例数据为 Iris 数据集（见 [data/iris.data.csv](data/iris.data.csv)）。

## 主要文件
- [SVM.py](SVM.py) — 包含类 `SVMClassifier`和相关训练 / 预测 / 可视化方法
- [kernel_func.py](kernel_func.py) — 提供核函数工厂：`kernel_func.linear kernel_func.poly kernel_func.rbf`
- [KNN.py](KNN.py) — 简单的 KNN 实现，类名为 `KNN`，包含训练与预测
- [bp.py](bp.py) — 三层 BP 神经网络实现，类名为 `BP`，包含训练与测试方法。
- [test_svm_classifier.py](test_svm_classifier.py) — 演示如何用 Iris 数据训练 / 可视化 SVM
- [data/iris.data.csv](data/iris.data.csv) — Iris 原始数据 CSV 文件。

## 依赖
- Python 3.6+
- numpy
- pandas
- matplotlib

## 安装示例
```sh
pip install numpy pandas matplotlib
```

## 运行示例
```sh
# bp
python .\bp.py
# KNN
py .\KNN.py
# SVM
python .\test_svm_classifier.py
```