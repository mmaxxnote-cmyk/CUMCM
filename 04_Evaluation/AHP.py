import numpy as np



class AHP:


    def __init__(
            self,
            matrix
    ):

        """
        matrix:
            判断矩阵
        """

        self.matrix=np.array(
            matrix,
            dtype=float
        )



    def calculate_weight(self):

        """
        特征向量法求权重
        """


        eigenvalues,eigenvectors=np.linalg.eig(
            self.matrix
        )


        # 最大特征值对应向量

        index=np.argmax(
            eigenvalues.real
        )


        weight=eigenvectors[:,index].real


        # 归一化

        weight=weight/weight.sum()


        return weight



    def consistency_check(
            self,
            weight
    ):

        """
        一致性检验
        """


        n=len(weight)


        eigenvalue=np.max(
            np.linalg.eigvals(
                self.matrix
            ).real
        )


        CI=(
            eigenvalue-n
        )/(n-1)


        # RI表

        RI={
            1:0,
            2:0,
            3:0.58,
            4:0.90,
            5:1.12,
            6:1.24,
            7:1.32,
            8:1.41,
            9:1.45
        }


        CR=CI/RI[n]


        return CR