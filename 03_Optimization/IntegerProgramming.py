import pulp

def integer_programming(
        c,
        A,
        b,
        maximize=True
):
    """
    整数规划求解

    参数:
        c:
            目标函数系数

        A:
            约束矩阵

        b:
            约束右侧

    返回:
        最优变量和目标值
    """


    n = len(c)


    # 创建问题

    if maximize:
        problem = pulp.LpProblem(
            "IntegerProgramming",
            pulp.LpMaximize
        )

    else:
        problem = pulp.LpProblem(
            "IntegerProgramming",
            pulp.LpMinimize
        )


    # 定义整数变量

    x = [
        pulp.LpVariable(
            f"x{i}",
            lowBound=0,
            cat="Integer"
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
        "x":result,
        "value":value
    }