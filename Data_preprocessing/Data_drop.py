
def column_drop(data, object_cols_drop):
    """
    丢弃指定名称的列，一般为缺值或者无意义的列
    :param data: 输入数据
    :param object_cols_drop: 需要丢弃的列
    :return: 丢弃列后的数据
    """
    data = data.drop(object_cols_drop, axis=1)
    return data

# # 测试
# from Data_read import data_read
# data = data_read()
# object_cols_drop = ['Car', 'BuildingArea', 'YearBuilt', 'CouncilArea']
# data_drop = column_drop(data, object_cols_drop)
#
# print(data_drop.head())
