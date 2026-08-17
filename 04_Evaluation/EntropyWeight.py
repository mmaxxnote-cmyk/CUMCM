import numpy as np



class EntropyWeight:


    def __init__(self):

        pass



    def normalize(self,X):

        """
        极差标准化
        """

        X=np.array(
            X,
            dtype=float
        )


        result=np.zeros_like(X)


        for j in range(X.shape[1]):

            col=X[:,j]


            result[:,j]=(
                col-col.min()
            )/(
                col.max()-col.min()
            )


        return result



    def calculate_weights(self,X):

        """
        计算熵权
        """


        X=self.normalize(X)



        # 防止log(0)

        X=X+1e-12



        # 比重

        P=X/X.sum(axis=0)



        n=X.shape[0]


        # 熵值

        E=(
            -1/np.log(n)
            *
            np.sum(
                P*np.log(P),
                axis=0
            )
        )


        # 差异系数

        D=1-E



        # 权重

        W=D/D.sum()


        return W