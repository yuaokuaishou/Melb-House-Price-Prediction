from sklearn.preprocessing import MinMaxScaler, StandardScaler


def data_normalization(data, object_cols_normalization):
    """
    数据归一化
    :param data:输入数据
    :param object_cols_normalization:需要二维数组
    :return:
    """
    min_max_scale = MinMaxScaler()
    for col in object_cols_normalization:
        data[col] = min_max_scale.fit_transform(data[col])
    return data


def data_scale(data, object_cols_scaler):
    """
    数据标准化
    :param data: 输入数据
    :param object_cols_scaler:
    :return:
    """
    standard_scalar = StandardScaler()
    for col in object_cols_scaler:
        data[col] = standard_scalar.fit_transform(data[col])
    return data

