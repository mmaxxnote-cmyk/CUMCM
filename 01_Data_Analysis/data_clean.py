import pandas as pd


def check_missing(data):
    """
    查看缺失值
    """
    return data.isnull().sum()



def fill_missing(data, method="mean"):
    """
    缺失值填充

    method:
        mean  平均值
        median 中位数
    """

    data = data.copy()

    if method == "mean":
        data = data.fillna(data.mean(numeric_only=True))

    elif method == "median":
        data = data.fillna(data.median(numeric_only=True))

    else:
        raise ValueError("未知填充方法")

    return data



def remove_duplicates(data):
    """
    删除重复数据
    """

    return data.drop_duplicates()



def describe_data(data):
    """
    查看数据统计信息
    """

    return data.describe()



