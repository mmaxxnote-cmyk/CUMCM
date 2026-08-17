import numpy as np
import random
import math



class SimulatedAnnealing:


    def __init__(
            self,
            objective_func,
            bounds,
            T=100,
            cooling_rate=0.95,
            iterations=1000
    ):

        """
        模拟退火

        objective_func:
            目标函数

        bounds:
            参数范围

        T:
            初始温度

        cooling_rate:
            降温速度

        iterations:
            迭代次数
        """


        self.objective_func = objective_func

        self.bounds = bounds

        self.T = T

        self.cooling_rate = cooling_rate

        self.iterations = iterations



    def random_solution(self):

        """
        随机生成一个解
        """

        return np.array(
            [
                np.random.uniform(low,high)
                for low,high in self.bounds
            ]
        )



    def neighbor(self,x):

        """
        产生邻域解
        """

        new_x=x.copy()


        index=random.randint(
            0,
            len(x)-1
        )


        low,high=self.bounds[index]


        new_x[index]=np.random.uniform(
            low,
            high
        )


        return new_x



    def run(self):


        current=self.random_solution()


        best=current.copy()


        current_value=self.objective_func(
            current
        )


        best_value=current_value


        T=self.T



        for _ in range(self.iterations):


            new=self.neighbor(
                current
            )


            new_value=self.objective_func(
                new
            )


            delta=new_value-current_value


            # 最大化问题

            if (
                delta>0
                or
                random.random()
                <
                math.exp(delta/T)
            ):


                current=new

                current_value=new_value



            if current_value>best_value:

                best=current.copy()

                best_value=current_value



            T*=self.cooling_rate



        return {
            "x":best,
            "value":best_value
        }