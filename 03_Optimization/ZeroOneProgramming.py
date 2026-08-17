import pulp

def zero_one_programming(
        c,
        A,
        b,
        maximize=True
):
    """
    0-1整数规划

    x只能取0或1

    参数:
        c:
            目标函数系数

        A:
            约束矩阵

        b:
            约束右侧

    返回:
        最优选择方案
    """


    n = len(c)


    # 创建问题

    if maximize:

        problem = pulp.LpProblem(
            "ZeroOneProgramming",
            pulp.LpMaximize
        )

    else:

        problem = pulp.LpProblem(
            "ZeroOneProgramming",
            pulp.LpMinimize
        )


    # 定义0-1变量

    x = [
        pulp.LpVariable(
            f"x{i}",
            cat="Binary"
        )
        for i in range(n)
    ]


    # 目标函数

    problem += pulp.lpSum(
        c[i]*x[i]
        for i in range(n)
    )


    # 添加约束

    for i in range(len(A)):

        problem += (
            pulp.lpSum(
                A[i][j]*x[j]
                for j in range(n)
            )
            <= b[i]
        )


    # 求解

    problem.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )


    result = [
        pulp.value(x[i])
        for i in range(n)
    ]


    value = pulp.value(
        problem.objective
    )


    return {
        "x": result,
        "value": value
    }
