
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SVM import SVMClassifier
 
data = pd.read_csv('./data/iris.data.csv', header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']   
# print(data)

X = data.iloc[0:150, 0:2].values
y = data.iloc[0:150, 4].values
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1 
y[y == 'Iris-virginica'] = 2
X_setosa, y_setosa = X[0:50], y[0:50]                     # Iris-setosa 4个特征
X_versicolor, y_versicolor = X[50:100], y[50:100]         # Iris-versicolor 4个特征
X_virginica, y_virginica = X[100:150], y[100:150]         # Iris-virginica 4个特征

# 训练集
X_setosa_train = X_setosa[:30, :]
y_setosa_train = y_setosa[:30]
X_versicolor_train = X_versicolor[:30, :]
y_versicolor_train = y_versicolor[:30]
X_virginica_train = X_virginica[:30, :]
y_virginica_train = y_virginica[:30]
X_train = np.vstack([X_setosa_train, X_versicolor_train, X_virginica_train])
y_train = np.hstack([y_setosa_train, y_versicolor_train, y_virginica_train])

# 测试集
X_setosa_test = X_setosa[30:50, :]
y_setosa_test = y_setosa[30:50]
X_versicolor_test = X_versicolor[30:50, :]
y_versicolor_test = y_versicolor[30:50]
X_virginica_test = X_virginica[30:50, :]
y_virginica_test = y_virginica[30:50]
X_test = np.vstack([X_setosa_test, X_versicolor_test, X_virginica_test])
y_test = np.hstack([y_setosa_test, y_versicolor_test, y_virginica_test])

# C = 1000，倾向于硬间隔
svm = SVMClassifier(C=1000)
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
 
plt.figure(figsize=(14, 10))
plt.subplot(221)
svm.plt_svm(X_train, y_train, is_show=False, is_margin=True)
plt.subplot(222)
svm.plt_loss_curve(is_show=False)
 
# C = 1，倾向于软间隔
svm = SVMClassifier(C=1)
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
 
plt.subplot(223)
svm.plt_svm(X_train, y_train, is_show=False, is_margin=True)
plt.subplot(224)
svm.plt_loss_curve(is_show=False)
 
plt.tight_layout()
plt.show()