from sklearn.model_selection import train_test_split

from Data_preprocessing.Data_drop import column_drop
from Data_preprocessing.Data_imputer import data_add_missing_value
from Data_preprocessing.Data_normalization import data_normalization, data_scale
from Data_preprocessing.Data_read import data_read
from Data_preprocessing.Data_text_to_number import text_to_number


def data_preprocessing():
    """
    :return: 预处理好的数据
    """
    data = data_read()
    # 丢弃列
    object_cols_drop = ['Address']
    data = column_drop(data, object_cols_drop)
    # 文本转换为数字
    object_cols_ton = ['Suburb', 'Type', 'CouncilArea', 'Method', 'SellerG', 'Regionname']
    data = text_to_number(data, object_cols_ton)
    # 补充列数据
    object_cols_impute = [['Car', 'BuildingArea', 'YearBuilt']]
    data = data_add_missing_value(data, object_cols_impute)
    # 数据归一化
    object_cols_normalization = [['Suburb', 'Rooms', 'Type', 'Method', 'SellerG',
                                  'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom',
                                  'Car', 'YearBuilt', 'Lattitude', 'Longtitude', 'Regionname',
                                  'Propertycount', 'CouncilArea']]
    data = data_normalization(data, object_cols_normalization)
    # 数据标准化
    object_cols_scaler = [['Landsize', 'BuildingArea']]
    data = data_scale(data, object_cols_scaler)

    return data


def data_split():
    """
    此函数可以进行全部的数据预处理
    :return: 将数据分成训练数据和验证数据
    """
    data = data_preprocessing()
    # 输出y和输入x
    y = data['Price']
    x = data.drop(['Price'], axis=1)
    # 将数据分成训练数据和验证数据
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, random_state=1)

    return train_x, valid_x, train_y,valid_y
