import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # K临近分类
from sklearn.model_selection import cross_val_score  # 交叉检验
from matplotlib import pyplot as plt

estimator = KNeighborsClassifier()
# 输出主目录位置
data_folder = os.path.expanduser("~")
data_filename = os.path.join(data_folder, "Ionosphere", "ionosphere.data")

X = np.zeros((351, 34), dtype="float")
Y = np.zeros((351,), dtype="bool")
# 创建csv阅读器对象
with open(data_filename, "r") as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        Y[i] = row[-1] == 'g'
# 创建训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=14)
estimator.fit(X_train, Y_train)
Y_predicted = estimator.predict(X_test)
accuracy = np.mean(Y_test == Y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))
scores = cross_val_score(estimator, X, Y, scoring="accuracy")
average_accuracy = np.mean(scores)*100
print("The average_accuracy is {0:.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, Y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

plt.plot(parameter_values, avg_scores, '-o')
plt.show()