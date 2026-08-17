import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def min_max_normalize(data):
    """
    Min-Max标准化

    将数据缩放到[0,1]
    """

    scaler = MinMaxScaler()

    result = scaler.fit_transform(data)

    return pd.DataFrame(
        result,
        columns=data.columns
    )



def z_score_normalize(data):
    """
    Z-score标准化

    均值0，方差1
    """

    scaler = StandardScaler()

    result = scaler.fit_transform(data)

    return pd.DataFrame(
        result,
        columns=data.columns
    )



def positive_negative_normalize(data, positive_columns, negative_columns):
    """
    指标正负向处理

    positive_columns:
        越大越好的指标

    negative_columns:
        越小越好的指标
    """

    result = data.copy()


    # 正向指标
    for col in positive_columns:
        result[col] = (
            result[col]-result[col].min()
        ) / (
            result[col].max()-result[col].min()
        )


    # 负向指标
    for col in negative_columns:
        result[col] = (
            result[col].max()-result[col]
        ) / (
            result[col].max()-result[col].min()
        )


    return result