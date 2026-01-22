from numpy import *
from matplotlib import pyplot as plt

def load_data_set(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet

def distance_euclidean(vector1, vector2):
    return sqrt(sum(power(vector1-vector2, 2))) 


def rand_center(dataSet, k):
    n = shape(dataSet)[1] 


    # 初始化质心
    centroids = asmatrix(zeros((k, n)))  
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids 


def k_means(dataSet,k,distMeas = distance_euclidean,creatCent = rand_center):
    m = shape(dataSet)[0]
    # 建立簇分配结果矩阵
    clusterAssment = asmatrix(zeros((m, 2)))
    centroids = creatCent(dataSet, k)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  
            minDist = inf 
            minIndex = -1 
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k): 
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


datMat = asmatrix(load_data_set('./data/testSet.txt'))
myCentroids, clusterAssing = k_means(datMat, 4)
plt.scatter(array(datMat)[:, 0], array(datMat)[:, 1], c=array(clusterAssing)[:, 0].T)
plt.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist(), c="r")
plt.show()