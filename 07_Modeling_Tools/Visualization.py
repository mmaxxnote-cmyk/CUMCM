import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



class Visualization:


    def line_plot(
            self,
            x,
            y,
            title="Line Plot"
    ):

        """
        折线图
        """

        plt.figure(
            figsize=(8,5)
        )


        plt.plot(
            x,
            y,
            marker="o"
        )


        plt.title(
            title
        )


        plt.xlabel(
            "X"
        )


        plt.ylabel(
            "Y"
        )


        plt.grid()


        plt.show()



    def scatter_plot(
            self,
            x,
            y,
            title="Scatter Plot"
    ):

        """
        散点图
        """


        plt.figure(
            figsize=(8,5)
        )


        plt.scatter(
            x,
            y
        )


        plt.title(
            title
        )


        plt.show()



    def heatmap(
            self,
            data,
            title="Heatmap"
    ):

        """
        热力图
        """


        plt.figure(
            figsize=(8,6)
        )


        sns.heatmap(
            data,
            annot=True,
            cmap="coolwarm"
        )


        plt.title(
            title
        )


        plt.show()



    def prediction_compare(
            self,
            y_true,
            y_pred
    ):

        """
        真实值预测值比较
        """


        plt.figure(
            figsize=(8,5)
        )


        plt.plot(
            y_true,
            label="True"
        )


        plt.plot(
            y_pred,
            label="Prediction"
        )


        plt.legend()


        plt.show()