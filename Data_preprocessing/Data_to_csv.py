from Data_preprocessing.Data_split_API import data_preprocessing


def data_output():
    data = data_preprocessing()
    data.to_csv('/Users/yuao/Downloads/melb_data_new.csv')
    return


data_output()
