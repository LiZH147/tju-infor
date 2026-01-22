项目使用 Cornell 电影评论数据集（review polarity）完成文本情感分类，包含朴素贝叶斯与基于决策树（ID3 / C4.5）的实现。
# 项目结构
data/
- txt_sentoken/
 - pos/ # 正面评论文本文件（1000）
 - neg/ # 负面评论文本文件（1000）
 - poldata.README.2.0
- Bayes.py # 朴素贝叶斯：数据加载、向量化、训练与测试
- DecisionTreeClassifier.py # 决策树实现（ID3 与 C4.5），使用 Bayes.py 中的数据处理函数
- README.md

# 功能概述
## Bayes.py
- loadData(): 读取 data/txt_sentoken 中文本并做简单正则清洗
- splitDataSet(): 随机划分训练集/测试集（默认 0.67）
- getVocab(), word2Vec(): 建立词表并将文档转为词频向量
- TrainBayes(), classifyBayes(): 训练与分类，计算错误率
## DecisionTreeClassifier.py
- 使用 Bayes.py 的数据加载与向量化函数
- 将词频转换为二值特征（出现/未出现）
- 为效率只选取训练集中出现频率最高的前 N 个词（默认 N=100）作为特征
- 支持 ID3 与 C4.5（信息增益 / 增益率）构建树并评估准确率/错误率

# 设计与注意事项
- 文本预处理为简单正则清洗，未做停用词或词形还原，可能影响性能。
- 决策树采用离散二值特征（出现/未出现），原始词频被二值化以适配 ID3/C4.5。
- 原始词表规模大，故只用 top N 高频词来构建树以节省时间与内存。
- 结果受随机划分影响，可以多次运行取均值或固定随机种子重现实验。

# 预期输出
- Bayes.py：输出测试集错误率（浮点数）。
- DecisionTreeClassifier.py：打印训练/测试集大小、词汇表大小、ID3 与 C4.5 的准确率与错误率对比。