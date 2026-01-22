import numpy as np
from scipy.spatial.distance import cdist
import random
import matplotlib.pyplot as plt
import copy

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def total_cost(dataMat, medoids):

    med_idx = medoids["cen_idx"]
    k = len(med_idx) 
    cost = 0
    medObject = dataMat[med_idx,:]
    dis = cdist(dataMat, medObject, 'euclidean')  
    cost = dis.min(axis=1).sum()
    medoids["t_cost"] = cost


def Assment(dataMat, mediods):
    med_idx = mediods["cen_idx"] 
    med = dataMat[med_idx] 
    k = len(med_idx) 

    dist = cdist(dataMat, med, 'euclidean')
    idx = dist.argmin(axis=1) 
    for i in range(k):
        mediods[i] = np.where(idx == i) 


def PAM(data, k):
    data = np.asmatrix(data)
    N = len(data) 
    cur_medoids = {}
    cur_medoids["cen_idx"] = random.sample(range(N), k)
    Assment(data, cur_medoids)
    total_cost(data, cur_medoids)
    old_medoids = {}
    old_medoids["cen_idx"] = []

    iter_counter = 1
    while not set(old_medoids['cen_idx']) == set(cur_medoids['cen_idx']):
        print("iteration counter:", iter_counter)
        iter_counter = iter_counter + 1
        best_medoids = copy.deepcopy(cur_medoids)
        old_medoids = copy.deepcopy(cur_medoids)
        for i in range(N):
            for j in range(k):
                if not i == j:  
                    tmp_medoids = copy.deepcopy(cur_medoids)
                    tmp_medoids["cen_idx"][j] = i

                    Assment(data, tmp_medoids)
                    total_cost(data, tmp_medoids)

                    if(best_medoids["t_cost"]>tmp_medoids["t_cost"]):
                        best_medoids = copy.deepcopy(tmp_medoids) 

        cur_medoids = copy.deepcopy(best_medoids) 
        print("current total cost is:",cur_medoids["t_cost"])
    return cur_medoids


def test():
    dim = 2
    N = 100
    d1 = np.random.normal(1, .2, (N, dim))
    d2 = np.random.normal(2, .5, (N, dim))
    d3 = np.random.normal(3, .3, (N, dim))
    data = np.vstack((d1, d2, d3))

    k = 3
    medoids = PAM(data, k)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8] 

    ax1 = fig.add_axes(rect, label='ax1', frameon=True)
    ax1.set_title('Clusters Result')
    ax1.scatter(data[medoids[0], 0], data[medoids[0], 1], c='r')
    ax1.scatter(data[medoids[1], 0], data[medoids[1], 1], c='g')
    ax1.scatter(data[medoids[2], 0], data[medoids[2], 1], c='y')
    ax1.scatter(data[medoids['cen_idx'], 0], data[medoids['cen_idx'], 1], marker='x', s=500)
    plt.show()


if __name__ =='__main__':
    test()