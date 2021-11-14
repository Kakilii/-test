import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset.data
Y = dataset.target
# 特征均值计算
attribute_means = X.mean(axis=0)

X_d = np.array(X >= attribute_means, dtype='int')


# OneR算法
# 根据待预测的数据的某项特征预测类别，并且给出错误率
def train_feature_value(X, y_ture, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_ture):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(),
                                 key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]

    incorrect_predictions = [class_counts for class_value, class_count in class_counts.items()
                             if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class, error


# 求总的错误率
def train_on_feature(X, y_true, feature_index):
    values = set(X[:,feature_index])
    predictors = {}  # 预测器
    errors = []  # 错误率
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


X_train, X_test, Y_train, Y_test = train_test_split(X_d, Y, random_state=14)
all_predictors = {}
errors = {}
for feature_index in range(X_train.shape[1]):
    predictors, total_error = train_on_feature(X_train, Y_train,
                                               feature_index)
    all_predictors[feature_index] = predictors
    errors[feature_index] = total_error
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
model = {'variable': best_variable,
         'predictor': all_predictors[best_variable]}


# 对遍历数据集中的每条数据完成预测
def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    Y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return Y_predicted


Y_predicted = predict(X_test, model)
accuracy = np.mean(Y_predicted == Y_test) * 100
print("The  test accuracy is {:.1f}%".format(accuracy))
