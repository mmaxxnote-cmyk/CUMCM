import numpy as np

from scipy.optimize import curve_fit



class CurveFitting:


    def __init__(
            self,
            model_func
    ):

        """
        model_func:
            自定义函数

        例如:
            a*x+b
            a*exp(b*x)
        """

        self.model_func=model_func

        self.params=None



    def fit(
            self,
            x,
            y,
            p0=None
    ):

        """
        曲线拟合
        """


        params,_=curve_fit(
            self.model_func,
            x,
            y,
            p0=p0
        )


        self.params=params



    def predict(
            self,
            x
    ):

        """
        根据拟合参数预测
        """


        return self.model_func(
            x,
            *self.params
        )