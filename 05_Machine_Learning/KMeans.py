import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



class KMeansModel:


    def __init__(
            self,
            n_clusters=3
    ):

        """
        n_clusters:
            聚类数量
        """


        self.scaler = StandardScaler()


        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )



    def fit_predict(self, X):

        """
        标准化 + 聚类
        """


        X_scaled = self.scaler.fit_transform(
            X
        )


        labels = self.model.fit_predict(
            X_scaled
        )


        return labels



    def centers(self):

        """
        查看聚类中心
        """

        return self.model.cluster_centers_