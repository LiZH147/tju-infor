import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

iris = pd.read_csv("data/iris.data.csv") 
df = iris  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  
features = columns[:len(columns) - 1] 
dataset = df[features]
attributes = len(df.columns) - 1 
original_labels = list(df[columns[-1]]) 

UNCLASSIFIED = 0
NOISE = -1


def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


def find_points_in_eps(point_id, eps, dists):
    index = (dists[point_id] <= eps)
    return np.where(index == True)[0].tolist()


# 聚类扩展
# dists ： 所有数据两两之间的距离  N x N
# labs :   所有数据的标签 labs N，
# cluster_id ： 一个簇的标号
# eps ： 密度评估半径
# seeds： 用来进行簇扩展的点
# min_points： 半径内最少的点数
def expand_cluster(dists, labs, cluster_id, seeds, eps, min_points):
    i = 0
    while i < len(seeds):
        Pn = seeds[i]
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        elif labs[Pn] == UNCLASSIFIED:
            labs[Pn] = cluster_id
            new_seeds = find_points_in_eps(Pn, eps, dists)

            if len(new_seeds) >= min_points:
                seeds = seeds + new_seeds

        i = i + 1


def dbscan(datas, Eps, MinPts):
    dists = getDistanceMatrix(datas)

    n_points = datas.shape[0]
    labs = [UNCLASSIFIED] * n_points

    cluster_id = 0
    for point_id in range(0, n_points):
        if not (labs[point_id] == UNCLASSIFIED):
            continue

        seeds = find_points_in_eps(point_id, Eps, dists)

        if len(seeds) < MinPts:
            labs[point_id] = NOISE
        else:
            cluster_id = cluster_id + 1
            labs[point_id] = cluster_id
            expand_cluster(dists, labs, cluster_id, seeds, Eps, MinPts)
    return labs, cluster_id


def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


def draw_cluster(datas, labs, n_cluster):
    plt.cla()
    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
    if attributes > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维
    # plt.scatter(datas[:, 0], datas[:, 1], s=7., color="black")
    # plt.show()
    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(datas[i, 0], datas[i, 1], s=7., color=(0, 0, 0))
        else:
            plt.scatter(datas[i, 0], datas[i, 1], s=7., color=colors[lab - 1])

    plt.show()


if __name__ == "__main__":
    # 设置DBSCAN参数
    Eps = 0.45
    MinPts = 4

    datas = np.array(dataset).astype(np.float32)
    datas = StandardScaler().fit_transform(datas)
    label, cluster_id = dbscan(datas, Eps=Eps, MinPts=MinPts)  
    print(original_labels) 
    print("label of my dbscan")
    print(label)  
    F_measure, ACC, NMI, RI, ARI = clustering_indicators(original_labels, label)  # 计算聚类指标
    print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI, "ARI", ARI)
    draw_cluster(datas, label, cluster_id) 


