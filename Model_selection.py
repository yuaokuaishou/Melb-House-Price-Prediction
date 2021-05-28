from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from time import *
from sklearn.metrics import mean_absolute_error

from Data_preprocessing.Data_split_API import data_split

import numpy as np
import matplotlib.pyplot as plt


def random_forest_regress_model():
    """
    随机森林模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = RandomForestRegressor(random_state=1)
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def support_vector_machine_model():
    """
    支持向量机模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = SVR()
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def ada_boost_regress_model():
    """
    AdaBoost回归模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss="linear")
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def gradient_boost_regress_model():
    """
    Gradient Boost回归模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def extra_trees_regress_model():
    """
    极端随机树回归--房价预测最佳
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = ExtraTreesRegressor(n_estimators=1000, criterion="mse", max_depth=None,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features="auto",
                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, bootstrap=False, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def bagging_regress_model():
    """
    Bagging回归模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = BaggingRegressor(base_estimator=None, n_estimators=1000, max_samples=1.0,
                             max_features=1.0, bootstrap=True, bootstrap_features=False,
                             oob_score=False, warm_start=False, random_state=None, verbose=0)
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data


def k_neighbors_regress_model():
    """
    K近邻回归模型
    :return:返回预测结果以及实际结果
    """
    train_x, valid_x, train_y, valid_y = data_split()
    model = KNeighborsRegressor(n_neighbors=5, weights="uniform", algorithm="auto",
                                leaf_size=30, p=2, metric="minkowski", metric_params=None)
    model.fit(train_x, train_y)
    model_prediction = model.predict(valid_x)

    # 统一预测结果与实际结果的格式
    predict_data = []
    for data in model_prediction:
        predict_data.append(data)

    real_data = []
    for data in valid_y:
        real_data.append(data)

    return predict_data, real_data

# 支持向量机模型测试

start = time()

# 选择对应模型--必选
# predict, real = random_forest_regress_model()
# predict, real = ada_boost_regress_model()
# predict, real = support_vector_machine_model()
# predict, real = gradient_boost_regress_model()
# predict, real = extra_trees_regress_model()
# predict, real = bagging_regress_model()
# predict, real = k_neighbors_regress_model()

end = time()

run_time = end - start
print('Running time is:', run_time)

error = []
for i in range(len(predict)):
    error.append(np.abs((predict[i] - real[i]) / real[i]))

print(np.mean(error))
print(mean_absolute_error(predict, real))

# 画图对比预测值和实际值
plt.rcParams['figure.figsize'] = (100.0, 8.0)

plt.plot(predict, color='black')
plt.plot(real, color='red')

plt.show()
