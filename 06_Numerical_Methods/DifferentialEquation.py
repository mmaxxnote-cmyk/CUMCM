import numpy as np

from scipy.integrate import solve_ivp



class DifferentialEquation:


    def __init__(
            self,
            func
    ):

        """
        func:
            微分方程

            dy/dt=f(t,y)
        """

        self.func=func



    def solve(
            self,
            t_span,
            y0,
            t_eval=None
    ):

        """
        求解ODE


        t_span:
            时间范围

        y0:
            初始条件

        t_eval:
            输出时间点
        """


        result=solve_ivp(
            self.func,
            t_span,
            y0,
            t_eval=t_eval
        )


        return {
            "t":result.t,
            "y":result.y
        }