import pandas as pd


def data_read():
    """
    数据读取
    :return: 返回读取到的数据
    """
    data = pd.read_csv('/Users/yuao/Downloads/melb_data.csv')
    return data


def check_missing_value(data):
    """
    仅用于寻找有缺值的列
    :param data: 读取到的数据
    :return: 有缺值的列名称
    """
    cols_with_missing_value = [col for col in data.columns if data[col].isnull().any()]
    print(cols_with_missing_value)
    return

# melb_data.csv中有缺值的列名称
# cols_with_missing_value = ['Car', 'BuildingArea', 'YearBuilt', 'CouncilArea']
