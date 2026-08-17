import numpy as np



class GM11:


    def __init__(self):

        self.a = None
        self.b = None



    def fit(self, data):

        """
        训练GM(1,1)

        data:
            一维时间序列
        """


        x0 = np.array(
            data,
            dtype=float
        )


        # 一次累加

        x1 = np.cumsum(x0)


        # 构造B矩阵

        z = (
            x1[:-1]
            +
            x1[1:]
        ) / 2


        B = np.column_stack(
            (
                -z,
                np.ones(len(z))
            )
        )


        Y = x0[1:]


        # 最小二乘求a,b

        params = np.linalg.inv(
            B.T @ B
        ) @ B.T @ Y


        self.a = params[0]

        self.b = params[1]



    def predict(self, steps):

        """
        预测未来steps个点
        """


        result=[]


        for k in range(steps):

            value = (
                self.b/self.a
                +
                (
                    1-
                    self.b/self.a
                )
                *
                np.exp(
                    -self.a*k
                )
            )


            result.append(value)


        return np.array(result)