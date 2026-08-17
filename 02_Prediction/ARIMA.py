import pandas as pd

from statsmodels.tsa.arima.model import ARIMA



class ARIMAModel:


    def __init__(self, order=(1,1,1)):

        """
        order:
            (p,d,q)

        """

        self.order = order

        self.model = None



    def fit(self, data):

        """
        训练模型

        data:
            时间序列
        """

        self.model = ARIMA(
            data,
            order=self.order
        )


        self.model = self.model.fit()



    def predict(self, steps):

        """
        预测未来steps步
        """

        forecast = self.model.forecast(
            steps=steps
        )


        return forecast



    def summary(self):

        """
        查看模型信息
        """

        return self.model.summary()