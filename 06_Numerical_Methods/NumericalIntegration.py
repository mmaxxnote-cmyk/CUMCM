import numpy as np

from scipy.integrate import quad



class NumericalIntegration:


    def trapezoid(
            self,
            x,
            y
    ):

        """
        梯形积分

        x:
            横坐标

        y:
            函数值
        """


        return np.trapz(
            y,
            x
        )



    def simpson(
            self,
            func,
            a,
            b
    ):

        """
        Simpson积分
        """


        from scipy.integrate import simpson


        x=np.linspace(
            a,
            b,
            100
        )


        y=func(x)


        return simpson(
            y,
            x=x
        )



    def adaptive(
            self,
            func,
            a,
            b
    ):

        """
        自适应积分
        """


        result,error=quad(
            func,
            a,
            b
        )


        return result