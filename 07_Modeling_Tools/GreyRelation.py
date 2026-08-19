import numpy as np



class GreyRelation:


    def __init__(
            self,
            rho=0.5
    ):

        """
        rho:
            分辨系数
            通常0.5
        """

        self.rho=rho



    def normalize(
            self,
            X
    ):

        """
        初值化处理
        """

        X=np.array(
            X,
            dtype=float
        )


        return X / X[0]



    def calculate(
            self,
            reference,
            factors
    ):

        """
        reference:
            参考序列

        factors:
            因素序列
        """


        reference=self.normalize(
            reference
        )


        factors=self.normalize(
            factors
        )


        diff=np.abs(
            factors-reference[:,None]
        )


        min_diff=diff.min()

        max_diff=diff.max()



        coefficient=(
            min_diff
            +
            self.rho*max_diff
        )/(
            diff
            +
            self.rho*max_diff
        )


        relation=coefficient.mean(
            axis=0
        )


        return relation