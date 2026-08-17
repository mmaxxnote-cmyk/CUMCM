import numpy as np
from scipy.optimize import linprog



def linear_programming(
        c,
        A,
        b,
        maximize=False
):
    """
    线性规划求解

    目标:
        max/min c*x

    参数:
        c:
            目标函数系数

        A:
            约束矩阵

        b:
            约束右侧

        maximize:
            是否最大化

    返回:
        最优解
    """


    c = np.array(c)

    if maximize:
        c = -c


    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=(0,None),
        method="highs"
    )


    if result.success:

        if maximize:
            value = -result.fun
        else:
            value = result.fun


        return {
            "x": result.x,
            "value": value
        }

    else:

        return None
