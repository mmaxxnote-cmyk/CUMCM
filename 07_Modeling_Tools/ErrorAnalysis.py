import numpy as np



class ErrorAnalysis:


    def __init__(
            self,
            y_true,
            y_pred
    ):

        """
        y_true:
            真实值

        y_pred:
            预测值
        """

        self.y_true=np.array(
            y_true,
            dtype=float
        )

        self.y_pred=np.array(
            y_pred,
            dtype=float
        )



    def MAE(self):

        """
        平均绝对误差
        """

        return np.mean(
            np.abs(
                self.y_true-self.y_pred
            )
        )



    def MSE(self):

        """
        均方误差
        """

        return np.mean(
            (
                self.y_true-self.y_pred
            )**2
        )



    def RMSE(self):

        """
        均方根误差
        """

        return np.sqrt(
            self.MSE()
        )



    def MAPE(self):

        """
        平均百分比误差
        """

        return np.mean(
            np.abs(
                (
                    self.y_true-self.y_pred
                )
                /
                self.y_true
            )
        )*100



    def R2(self):

        """
        决定系数
        """

        ss_res=np.sum(
            (
                self.y_true-self.y_pred
            )**2
        )


        ss_tot=np.sum(
            (
                self.y_true-
                np.mean(self.y_true)
            )**2
        )


        return 1-ss_res/ss_tot



    def report(self):

        """
        综合报告
        """

        return {

            "MAE":self.MAE(),

            "MSE":self.MSE(),

            "RMSE":self.RMSE(),

            "MAPE":self.MAPE(),

            "R2":self.R2()

        }