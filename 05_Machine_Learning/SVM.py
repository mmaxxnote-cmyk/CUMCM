from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score



class SVMModel:


    def __init__(
            self,
            kernel="rbf",
            C=1
    ):

        """
        SVM分类模型

        kernel:
            核函数

        C:
            惩罚参数
        """


        self.scaler = StandardScaler()


        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True
        )



    def fit(self, X, y):

        """
        训练
        """

        X_scaled = self.scaler.fit_transform(
            X
        )


        self.model.fit(
            X_scaled,
            y
        )



    def predict(self, X):

        """
        分类预测
        """

        X_scaled = self.scaler.transform(
            X
        )


        return self.model.predict(
            X_scaled
        )



    def evaluate(self, X, y):

        """
        准确率
        """

        y_pred=self.predict(X)


        return accuracy_score(
            y,
            y_pred
        )