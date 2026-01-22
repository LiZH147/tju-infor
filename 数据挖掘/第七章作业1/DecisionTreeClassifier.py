import os
import Bayes
import operator
from math import log

def extractFeatures(trainingData, traingLabel, testData, testLabel, vocab):
    """
    提取特征，将文本转换为特征向量
    为了处理决策树，将词频转换为离散特征（出现/未出现）
    """
    trainingFeatures = []
    testFeatures = []
    
    # 转换为列表便于处理
    vocab_list = list(vocab)
    
    # 构建训练集特征
    for i, doc in enumerate(trainingData):
        vec = Bayes.word2Vec(vocab_list, doc)
        binary_vec = [1 if count > 0 else 0 for count in vec]
        binary_vec.append(traingLabel[i])
        trainingFeatures.append(binary_vec)
    
    # 构建测试集特征
    for i, doc in enumerate(testData):
        vec = Bayes.word2Vec(vocab_list, doc)
        binary_vec = [1 if count > 0 else 0 for count in vec]
        binary_vec.append(testLabel[i])
        testFeatures.append(binary_vec)
    
    return trainingFeatures, testFeatures, vocab_list

def calcShannonEnt(dataSet):
    """计算信息熵"""
    numEntries = len(dataSet)
    labelCounts = {}
    
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt


def splitDataSetByFeature(dataSet, axis, value):
    """按照特定特征值分割数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplitID3(dataSet):
    """ID3算法：根据信息增益选择最优特征"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSetByFeature(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    
    return bestFeature


def chooseBestFeatureToSplitC45(dataSet):
    """C4.5算法：根据增益率选择最优特征"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestGainRatio = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSetByFeature(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo -= prob * log(prob, 2)
        
        infoGain = baseEntropy - newEntropy
        
        if splitInfo != 0.0:
            gainRatio = infoGain / splitInfo
            if gainRatio > bestGainRatio:
                bestGainRatio = gainRatio
                bestFeature = i
    
    return bestFeature


def majorityCnt(classList):
    """返回出现次数最多的类别"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTreeID3(dataSet, labels):
    """用ID3算法构建决策树"""
    classList = [example[-1] for example in dataSet]
    
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplitID3(dataSet)
    
    if bestFeat == -1:
        return majorityCnt(classList)
    
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTreeID3(splitDataSetByFeature(dataSet, bestFeat, value), subLabels)
    
    return myTree


def createTreeC45(dataSet, labels):
    """用C4.5算法构建决策树"""
    classList = [example[-1] for example in dataSet]
    
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplitC45(dataSet)
    
    if bestFeat == -1:
        return majorityCnt(classList)
    
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTreeC45(splitDataSetByFeature(dataSet, bestFeat, value), subLabels)
    
    return myTree


def classifyByTree(inputTree, featLabels, testVec):
    """使用决策树进行分类"""
    try:
        firstStr = list(inputTree.keys())[0]
    except:
        return inputTree
    
    secondDict = inputTree[firstStr]
    
    if firstStr not in featLabels:
        return majorityCnt([secondDict[key] for key in secondDict.keys() if isinstance(secondDict[key], int)])
    
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    
    if key not in secondDict:
        return majorityCnt([secondDict[k] for k in secondDict.keys() if isinstance(secondDict[k], int)])
    
    valueOfFeat = secondDict[key]
    
    if isinstance(valueOfFeat, dict):
        classLabel = classifyByTree(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    
    return classLabel


def evaluateTree(tree, testData, vocabList):
    """评估决策树性能"""
    error_count = 0
    correct_count = 0
    
    for sample in testData:
        testVec = sample[:-1]
        true_label = sample[-1]
        
        predicted_label = classifyByTree(tree, vocabList[:len(testVec)], testVec)
        
        if predicted_label == true_label:
            correct_count += 1
        else:
            error_count += 1
    
    accuracy = correct_count / float(len(testData))
    error_rate = error_count / float(len(testData))
    
    return accuracy, error_rate

def main():
    print("=" * 60)
    print("电影评论情感分析 - 决策树分类")
    print("=" * 60)
    
    print("\n[1] 加载数据中...")
    pathDirPos = os.listdir("data/txt_sentoken/pos")
    pathDirNeg = os.listdir("data/txt_sentoken/neg")
    
    trainingData, traingLabel, testData, testLabel = Bayes.splitDataSet(pathDirPos, pathDirNeg, 0.67)
    print(f"训练集大小: {len(trainingData)}")
    print(f"测试集大小: {len(testData)}")
    
    print("\n[2] 构建词汇表...")
    vocab = Bayes.getVocab(trainingData)
    print(f"词汇表大小: {len(vocab)}")
    
    print("\n[3] 提取特征...")
    trainingFeatures, testFeatures, vocabList = extractFeatures(
        trainingData, traingLabel, testData, testLabel, vocab
    )
    print(f"特征维度: {len(vocabList)}")
    
    print("\n[4] 选择前100个最频繁的词汇进行决策树构建...")
    
    wordFreq = {}
    for doc in trainingData:
        for word in doc:
            wordFreq[word] = wordFreq.get(word, 0) + 1
    
    topWords = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)[:100]
    topWordList = [word[0] for word in topWords]
    
    trainingFeatures_top = []
    testFeatures_top = []
    
    for i, doc in enumerate(trainingData):
        vec = [1 if word in doc else 0 for word in topWordList]
        vec.append(traingLabel[i])
        trainingFeatures_top.append(vec)
    
    for i, doc in enumerate(testData):
        vec = [1 if word in doc else 0 for word in topWordList]
        vec.append(testLabel[i])
        testFeatures_top.append(vec)
    
    feature_labels = [f"word_{i}" for i in range(len(topWordList))]
    
    # 使用ID3算法构建决策树
    print("\n[5] 使用ID3算法构建决策树...")
    labels_id3 = feature_labels[:]
    tree_id3 = createTreeID3(trainingFeatures_top, labels_id3)
    accuracy_id3, error_rate_id3 = evaluateTree(tree_id3, testFeatures_top, feature_labels)
    print(f"ID3 - 准确率: {accuracy_id3:.4f}")
    print(f"ID3 - 错误率: {error_rate_id3:.4f}")
    
    # 使用C4.5算法构建决策树
    print("\n[6] 使用C4.5算法构建决策树...")
    labels_c45 = feature_labels[:]
    tree_c45 = createTreeC45(trainingFeatures_top, labels_c45)
    accuracy_c45, error_rate_c45 = evaluateTree(tree_c45, testFeatures_top, feature_labels)
    print(f"C4.5 - 准确率: {accuracy_c45:.4f}")
    print(f"C4.5 - 错误率: {error_rate_c45:.4f}")
    
    # 结果对比
    print("\n" + "=" * 60)
    print("算法对比结果")
    print("=" * 60)
    print(f"{'算法':<10} {'准确率':<10} {'错误率':<10}")
    print("-" * 60)
    print(f"{'ID3':<10} {accuracy_id3:<10.4f} {error_rate_id3:<10.4f}")
    print(f"{'C4.5':<10} {accuracy_c45:<10.4f} {error_rate_c45:<10.4f}")
    print("=" * 60)
    
    return tree_id3, tree_c45, feature_labels


if __name__ == "__main__":
    tree_id3, tree_c45, feature_labels = main()