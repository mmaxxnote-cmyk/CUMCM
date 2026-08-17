import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



class PCAModel:


    def __init__(
            self,
            n_components=2
    ):

        """
        n_components:
            保留多少个主成分
        """


        self.scaler = StandardScaler()


        self.pca = PCA(
            n_components=n_components
        )



    def fit_transform(self, X):

        """
        标准化 + PCA降维
        """


        X_scaled = self.scaler.fit_transform(
            X
        )


        X_pca = self.pca.fit_transform(
            X_scaled
        )


        return X_pca



    def explained_variance(self):

        """
        查看信息保留比例
        """

        return self.pca.explained_variance_ratio_