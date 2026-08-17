import numpy as np
from scipy.optimize import minimize



def nonlinear_optimization(
        objective_func,
        x0,
        bounds=None,
        constraints=None
):
    """
    非线性优化

    参数:

    objective_func:
        目标函数

    x0:
        初始解

    bounds:
        变量范围

    constraints:
        约束条件


    返回:
        最优结果
    """


    result = minimize(
        objective_func,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )


    if result.success:

        return {
            "x": result.x,
            "value": result.fun
        }

    else:

        return None

