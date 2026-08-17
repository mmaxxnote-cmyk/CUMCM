import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



class PolynomialRegressionModel:


    def __init__(self, degree=2):

        """
        degree:
            多项式次数
            例如:
            degree=2
            y=ax²+bx+c
        """

        self.degree = degree


        self.poly = PolynomialFeatures(
            degree=degree
        )


        self.model = LinearRegression()



    def fit(self, X, y):

        """
        训练
        """

        X_poly = self.poly.fit_transform(
            X
        )


        self.model.fit(
            X_poly,
            y
        )



    def predict(self, X):

        """
        预测
        """

        X_poly = self.poly.transform(
            X
        )


        return self.model.predict(
            X_poly
        )



    def evaluate(self, X, y):

        """
        评价
        """

        y_pred = self.predict(X)


        mse = mean_squared_error(
            y,
            y_pred
        )


        r2 = r2_score(
            y,
            y_pred
        )


        return {
            "MSE":mse,
            "R2":r2
        }