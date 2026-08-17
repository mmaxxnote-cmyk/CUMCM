import numpy as np



class MonteCarlo:


    def __init__(
            self,
            samples=10000
    ):

        """
        samples:
            随机样本数量
        """

        self.samples=samples



    def estimate_pi(self):

        """
        蒙特卡洛估计圆周率
        """


        x=np.random.uniform(
            -1,
            1,
            self.samples
        )


        y=np.random.uniform(
            -1,
            1,
            self.samples
        )


        inside=(
            x**2+y**2<=1
        )


        pi=4*np.sum(
            inside
        )/self.samples


        return pi



    def integration(
            self,
            func,
            a,
            b
    ):

        """
        蒙特卡洛积分

        计算:
        ∫f(x)dx
        """


        x=np.random.uniform(
            a,
            b,
            self.samples
        )


        y=func(x)


        result=(
            b-a
        )*np.mean(y)


        return result