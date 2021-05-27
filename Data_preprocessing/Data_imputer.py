from sklearn.impute import SimpleImputer


def data_add_missing_value(data, object_cols_impute):
    """
    用该列的均值补充缺失的值
    :param data: 输入数据
    :param object_cols_impute: 需要二维数组
    :return:
    """
    data_impute = SimpleImputer()
    for col in object_cols_impute:
        data_complete = data_impute.fit_transform(data[col])
        data[col] = data_complete
    return data

# # 测试
# from Data_read import data_read
# data = data_read()
# object_cols_impute = [['Car', 'BuildingArea', 'YearBuilt']]
# data_complete = data_add_missing_value(data, object_cols_impute)
# print(data_complete['YearBuilt'])
