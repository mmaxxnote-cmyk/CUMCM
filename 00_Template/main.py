"""
CUMCM 数学建模项目模板

流程：
1. 数据读取
2. 数据预处理
3. 模型建立
4. 模型评价
5. 结果保存
"""

import pandas as pd


def load_data(path):
    """
    数据读取
    """
    data = pd.read_excel(path)
    return data


def preprocess(data):
    """
    数据预处理
    """
    # 缺失值处理
    # 标准化
    # 异常值处理
    
    return data


def build_model(data):
    """
    建立模型
    """

    # 在这里调用：
    # 预测模型
    # 优化模型
    # 评价模型
    
    result = None

    return result


def evaluate(result):
    """
    模型评价
    """

    # 误差分析
    # 敏感性分析
    # 模型比较

    pass


def save_result(result):
    """
    保存结果
    """

    # 保存到 result 文件夹

    pass


def main():

    # 1. 数据读取
    data = load_data(
        "data/data.xlsx"
    )


    # 2. 数据处理
    data = preprocess(data)


    # 3. 模型建立
    result = build_model(data)


    # 4. 模型评价
    evaluate(result)


    # 5. 保存结果
    save_result(result)



if __name__ == "__main__":
    main()