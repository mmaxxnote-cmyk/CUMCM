from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



class DecisionTreeModel:


    def __init__(
            self,
            max_depth=None
    ):

        """
        决策树分类模型

        max_depth:
            最大深度
        """


        self.model = DecisionTreeClassifier(
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
        预测类别
        """

        return self.model.predict(
            X
        )



    def evaluate(self, X, y):

        """
        准确率
        """

        y_pred=self.predict(
            X
        )


        return accuracy_score(
            y,
            y_pred
        )



    def feature_importance(self):

        """
        特征重要性
        """

        return self.model.feature_importances_