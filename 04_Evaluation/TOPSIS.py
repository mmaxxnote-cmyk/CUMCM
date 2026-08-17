import numpy as np



class TOPSIS:


    def __init__(
            self,
            weights=None
    ):

        """
        TOPSIS评价

        weights:
            指标权重
        """

        self.weights=weights



    def normalize(self,X):

        """
        向量标准化
        """

        return X / np.sqrt(
            np.sum(X**2,axis=0)
        )



    def evaluate(
            self,
            X,
            positive
    ):

        """
        X:
            评价矩阵

        positive:
            指标类型

            1 正向指标
            0 负向指标
        """


        X=np.array(X,dtype=float)


        # 标准化

        Z=self.normalize(X)



        # 权重

        if self.weights is None:

            W=np.ones(
                Z.shape[1]
            )/Z.shape[1]

        else:

            W=np.array(
                self.weights
            )


        Z=Z*W



        # 理想最好

        best=[]

        worst=[]


        for i,p in enumerate(positive):

            if p==1:

                best.append(
                    max(Z[:,i])
                )

                worst.append(
                    min(Z[:,i])
                )

            else:

                best.append(
                    min(Z[:,i])
                )

                worst.append(
                    max(Z[:,i])
                )


        best=np.array(best)

        worst=np.array(worst)



        # 距离

        D_plus=np.sqrt(
            np.sum(
                (Z-best)**2,
                axis=1
            )
        )


        D_minus=np.sqrt(
            np.sum(
                (Z-worst)**2,
                axis=1
            )
        )


        # 综合评分

        score=D_minus/(D_plus+D_minus)


        return score