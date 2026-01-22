import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import random

# 加载数据
def loadData(filename):
    dataSet = np.loadtxt(filename, delimiter=' ', encoding="UTF-8")
    return dataSet
    
class AGNES():
    def __init__(self, data, cluster = 2):
        self.cluster = cluster
        self.data = data
        self.distance_matrix = []
        self.dic = {}
        self.dic_ = {}
        self.index = ['A', 'B', 'C', 'D', 'E']
        self.columns = ['A', 'B', 'C', 'D', 'E']
    
    def init_data(self, data, dic):
        dic={i:[chr(ord('A')+i)] for i in range(len(data))}
        
        data = self.calculate_distance(data)
        self.distance_matrix = data.copy()
        
        row, col = np.diag_indices_from(data) 
        temp = data.max() + 1
        data[row, col] = temp
        row_, col_ = np.triu_indices_from(data, k = 0)
        data[row_, col_] = temp
        
        return data, dic
                
    def train(self, cluster, method='train'):
        data = self.data.copy()
        dic = {}
        data, dic = self.init_data(data, dic)
        
        k = 0
        while k < len(data) - cluster:
            location = np.where(data == data.min()) # 找到此时矩阵距离最小值的坐标
            x, y = location[0][0], location[1][0]   # 分别获取横纵坐标
            x_ = self.index[x]
            y_ = self.columns[y]                    # 获取对应样本信息
            x_key = '-'
            y_key = '-'
            for key, value in dic.items():
                if x_ in value:
                    x_key = key
                if y_ in value:
                    y_key = key
                    
            dic[y_key].extend(dic[x_key])
            dic.pop(x_key)
            
            slic = dic[y_key]  # 更新簇的样本
            num = len(dic[y_key]) # 簇内样本的数目
            data_sum = np.zeros(5)
            for item in slic:
                data_sum += data[:, self.index.index(item)]
            data_sum /= num
            for item in slic:
            #   data[index.index(item)]=data_sum
                data[:, self.index.index(item)] = data_sum
                row, col = np.diag_indices_from(data) 
                temp = data.max() + 999
                data[row, col] = temp
                row_,col_ = np.triu_indices_from(data, k=0)
                data[row_, col_] = temp
            k += 1
        
        if method == 'train':
            self.dic = dic
        else:
            self.dic_ = dic
            
    def draw(self):
        dists = squareform(self.distance_matrix)
        linkage_type = 'average'   # single,complete,average不同方式
        linkage_matrix = linkage(dists, linkage_type)
        dendrogram(linkage_matrix, labels = self.index)
        plt.show()
        
    def process(self):
        for cluster in range(1, self.distance_matrix.shape[0] + 1):
            self.train(cluster, 'other')
            print('簇数：', cluster, self.dic_)
            
    def calculate_distance(self, data):
        distance_matrix = np.zeros((data.shape[0],data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                distance_matrix[i][j] = np.sum((data[i] - data[j])**2)
        return distance_matrix
# data = np.random.rand(5, 5)
# print(data)
data = loadData('./data/testSetAGNES.txt')
print(data)

model = AGNES(data)
model.train(cluster = 2)
model.process()
model.draw()