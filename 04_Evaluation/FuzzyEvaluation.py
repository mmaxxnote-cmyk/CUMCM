import numpy as np



class FuzzyEvaluation:


    def __init__(
            self,
            weights,
            relation_matrix
    ):

        """
        weights:
            指标权重

        relation_matrix:
            隶属度矩阵
        """


        self.weights=np.array(
            weights
        )


        self.R=np.array(
            relation_matrix
        )



    def evaluate(self):

        """
        模糊综合评价
        """


        result = self.weights @ self.R


        return result



    def grade(
            self,
            result,
            levels
    ):

        """
        判断最终等级
        """


        index=np.argmax(
            result
        )


        return levels[index]