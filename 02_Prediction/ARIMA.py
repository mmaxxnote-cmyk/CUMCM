import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score



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



    def evaluate(self, y_true, y_pred):

        """
        根据真实值和预测值计算评价指标。
        """

        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }



    def summary(self):

        """
        查看模型信息
        """

        return self.model.summary()
