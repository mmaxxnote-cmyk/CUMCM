import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data, column):
    """
    绘制单变量直方图

    参数:
        data: DataFrame
        column: 列名
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[column],
        bins=20
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"{column} Distribution")

    plt.show()



def plot_boxplot(data, column):
    """
    绘制箱线图
    用于观察异常值
    """

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        y=data[column]
    )

    plt.title(f"{column} Boxplot")

    plt.show()



def plot_scatter(data, x, y):
    """
    绘制散点图

    用于观察两个变量关系
    """

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data[x],
        data[y]
    )

    plt.xlabel(x)
    plt.ylabel(y)

    plt.title(
        f"{x} vs {y}"
    )

    plt.show()



def plot_line(x, y, xlabel="X", ylabel="Y"):
    """
    绘制折线图

    常用于时间序列
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()



def plot_heatmap(corr):
    """
    绘制相关性热力图
    """

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.show()



def plot_prediction_result(y_true, y_pred):
    """
    绘制预测值和真实值比较
    """

    plt.figure(figsize=(8,5))

    plt.plot(
        y_true,
        label="True"
    )

    plt.plot(
        y_pred,
        label="Prediction"
    )

    plt.legend()

    plt.title(
        "Prediction Result"
    )

    plt.show()