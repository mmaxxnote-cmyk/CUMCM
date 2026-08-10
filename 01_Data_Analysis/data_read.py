import pandas as pd


def read_data(file_path):
    """
    读取数据文件

    参数:
        file_path: 文件路径

    返回:
        pandas.DataFrame
    """

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)

    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)

    else:
        raise ValueError("不支持的文件格式")

    return data


if __name__ == "__main__":

    file = "data/example.xlsx"

    df = read_data(file)

    print(df.head())