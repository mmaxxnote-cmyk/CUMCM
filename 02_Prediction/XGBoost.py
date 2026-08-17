from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score



class XGBoostModel:


    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
    ):

        """
        XGBoost模型

        n_estimators:
            树数量

        learning_rate:
            学习率

        max_depth:
            树深度
        """


        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )



    def fit(self, X, y):

        """
        模型训练
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



    def evaluate(self, X, y):

        """
        模型评价
        """

        y_pred=self.predict(X)


        mse=mean_squared_error(
            y,
            y_pred
        )


        r2=r2_score(
            y,
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