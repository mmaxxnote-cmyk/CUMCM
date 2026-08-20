from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



class RandomForestModel:


    def __init__(
            self,
            n_estimators=100,
            max_depth=None
    ):

        """
        随机森林

        n_estimators:
            树的数量

        max_depth:
            树最大深度
        """


        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )



    def fit(self, X, y):

        """
        训练
        """

        self.model.fit(
            X,
            y
        )



    def predict(self, X):

        """
        预测
        """

        return self.model.predict(
            X
        )



    def evaluate(self, y_true, y_pred):

        """
        根据真实值和预测值评价模型。
        """

        mse=mean_squared_error(
            y_true,
            y_pred
        )


        r2=r2_score(
            y_true,
            y_pred
        )


        return {
            "MSE":mse,
            "R2":r2
        }



    def feature_importance(self):

        """
        特征重要性
        """

        return self.model.feature_importances_
