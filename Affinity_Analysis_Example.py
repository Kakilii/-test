import numpy as np
# 调用集合库，在没有对应键值的时候返回默认值
from collections import defaultdict
from operator import itemgetter

# 输出函数
def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}"
          .format(premise_name, conclusion_name))
    print(" - Support: {0}".format(support[premise, conclusion]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))


features = ["bread", "milk", "cheese", "apple", "banana"]
dataset_filename = "affinity_dataset.txt"

X = np.loadtxt(dataset_filename)
# 顺序是面包，牛奶，奶酪，苹果和香蕉
# 1表示购买，0表示未购买
# print(X[:5]) 打印前五排

# 获取多少人购买苹果
# num_apple_purchases = 0
# for sample in X:
#     if sample[3] == 1:
#         num_apple_purchases += 1
# print("{0} people bought Apples".format(num_apple_purchases))
valid_rules = defaultdict(int)  # 规则应验
invalid_rules = defaultdict(int)  # 规则无效
num_occurances = defaultdict(int)  # 条件相同的规则数量
# 样本数和特征数
n_sample, n_features = X.shape

for sample in X:
    for premise in range(5):
        if sample[premise] == 0: continue
        num_occurances[premise] += 1
        # 如果条件与结论相同，就跳过
        for conclusion in range(n_features):
            if premise == conclusion: continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1

support = valid_rules  # 支持度

confidence = defaultdict(float)  # 置信度
for premise, conclusion in valid_rules.keys():
    rule = (premise, conclusion)
    confidence[rule] = valid_rules[rule] / num_occurances[premise]

# 测试
# premise = 1
# conclusion = 3
# print_rule(premise, conclusion, support, confidence, features)
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

print("\n")

sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)