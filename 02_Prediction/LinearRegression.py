import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



class LinearRegressionModel:


    def __init__(self):

        self.model = LinearRegression()



    def fit(self, X, y):
        """
        训练模型
        """

        self.model.fit(
            X,
            y
        )



    def predict(self, X):
        """
        预测
        """

        return self.model.predict(X)



    def evaluate(self, y_true, y_pred):

        """
        根据真实值和预测值评价模型。
        """

        mse = mean_squared_error(
            y_true,
            y_pred
        )


        r2 = r2_score(
            y_true,
            y_pred
        )


        return {
            "MSE":mse,
            "R2":r2
        }
