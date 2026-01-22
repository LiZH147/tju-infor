
import os
import re
from numpy import *
import random
from numpy import *

'''获取数据，并去除数据中的多余符号,返回list类型的数据集'''
def loadData(pathDirPos,pathDirNeg):
    posDatas = []  # 积极评论
    negDatas = []  # 消极评论
    # 积极评论
    for allDir in pathDirPos:
        lineDataPos = []
        child = os.path.join('%s' % allDir)
        filename = r"data/txt_sentoken/pos/" + child
        with open(filename) as childFile:
            for lines in childFile:
                lineString = re.sub("[\n\.\!\/_\-$%^*(+\"\')]+|[+—()?【】“”！:,;.？、~@#￥%…&*（）0123456789]+", " ", lines)
                line = lineString.split(' ')          
                for strc in line:
                    if strc != "" and len(strc) > 1: 
                        lineDataPos.append(strc)
                posDatas.append(lineDataPos)
    # 消极评论
    for allDir in pathDirNeg:
        lineDataNeg = []
        child = os.path.join('%s' % allDir)
        filename = r"data/txt_sentoken/neg/" + child
        with open(filename) as childFile:
            for lines in childFile:
                lineString = re.sub("[\n\.\!\/_\-$%^*(+\"\')]+|[+—()?【】“”！:,;.？、~@#￥%…&*（）0123456789]+", " ", lines)
                line = lineString.split(' ')
                for strc in line:
                    if strc != "" and len(strc) > 1:
                        lineDataNeg.append(strc)
                negDatas.append(lineDataNeg)
    return posDatas,negDatas

def splitDataSet(pathDirPos,pathDirNeg,splitPara):
    trainingData=[]
    testData=[]
    traingLabel=[]
    testLabel=[]
    posData,negData=loadData(pathDirPos,pathDirNeg)
    pos_len=int(len(posData)/100)
    neg_len=int(len(negData)/100)
    #操作积极评论数据
    for i in range(pos_len):
        if(random.random()<splitPara):
            trainingData.append(posData[i])
            traingLabel.append(1)
        else:
            testData.append(posData[i])
            testLabel.append(1)
    for j in range(neg_len):
        if(random.random()<splitPara):
            trainingData.append(negData[j])
            traingLabel.append(0)
        else:
            testData.append(negData[j])
            testLabel.append(0)
    return trainingData,traingLabel,testData,testLabel

'''获取文本中的所有词汇，不重复'''
def getVocab(dataSet):
    dataVec=[]
    lenData=len(dataSet)
    for i in range(lenData):
        dataVec.extend(dataSet[i])
    vocab=set(dataVec)
    return vocab

'''将待处理文本转化为数值向量'''
def word2Vec(Vocablist,wordData):
    Vocablist=list(Vocablist)
    lenWordData=len(Vocablist)
    mathVec=zeros(lenWordData)
    for word in wordData:
        if word in Vocablist:
            mathVec[Vocablist.index(word)]+=1
    return mathVec

'''Bayes分类器训练函数'''
def TrainBayes(trainingData,trainingLabel,Vocablist):
    numfile=len(trainingData)
    P1=sum(trainingLabel)/float(numfile)       #类型为1的概率
    numWords=len(Vocablist)
    p1num=ones(numWords);p0num=ones(numWords)
    p1Denom=2.0;p0Denom=2.0                    #为了避免0概率的出现
    for i in range(numfile):
        if(trainingLabel[i]==1):
            p1num+=word2Vec(Vocablist,trainingData[i])
            p1Denom+=sum(word2Vec(Vocablist,trainingData[i]))
        else:
            p0num+=word2Vec(Vocablist,trainingData[i])
            p0Denom+=sum(word2Vec(Vocablist,trainingData[i]))
    p1A=log(p1num/float(p1Denom))
    p0A=log(p0num/float(p0Denom))
    return p0A,p1A,P1

'''Bayes分类函数'''
def classifyBayes(testVec,p0A,p1A,P1):
    class1=sum(testVec*p1A)+log(P1)
    class0=sum(testVec*p0A)+log(P1)
    if class1>class0:
        return 1
    else:
        return 0
    
def main():
    pathDirPos=os.listdir("data/txt_sentoken/pos")
    pathDirNeg=os.listdir("data/txt_sentoken/neg")
    trainingData,traingLabel,testData,testLabel=splitDataSet(pathDirPos,pathDirNeg,0.67)
    # print(trainingData,traingLabel)
    vocab=getVocab(trainingData)
    p0A,p1A,p1=TrainBayes(trainingData,traingLabel,vocab)
    len_testData=len(testData)
    '''分类并计算错误率'''
    error_count=0
    for i in range(len_testData):
        mathVec=word2Vec(vocab,testData[i])
        result=classifyBayes(mathVec,p0A,p1A,p1)
        if(result!=testLabel[i]):
            error_count+=1
    error_rate=error_count/float(len_testData)
    print(error_rate)

if __name__ == "__main__":
    main()