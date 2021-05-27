from sklearn import preprocessing


def text_to_number(data, object_cols_ton):
    """
    依照ASCII把字符串编码为数字
    :param data:导入的数据
    :param object_cols_ton:需要编码列的列表
    :return:编好码的数据
    """
    ton_encoder = preprocessing.LabelEncoder()
    for col in object_cols_ton:
        data_number_middle = ton_encoder.fit_transform(data[col])
        data[col] = data_number_middle
    return data

# # 测试
# from Data_read import data_read
# data = data_read()
# object_cols_ton = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Regionname']
# data_number = text_to_number(data, object_cols_ton)
# print(data_number.head())
