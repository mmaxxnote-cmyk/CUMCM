import numpy as np
import random



class GeneticAlgorithm:


    def __init__(
        self,
        fitness_func,
        bounds,
        population_size=50,
        generations=100,
        mutation_rate=0.1
    ):

        """
        遗传算法

        参数:

        fitness_func:
            目标函数

        bounds:
            每个变量范围

        population_size:
            种群大小

        generations:
            迭代次数

        mutation_rate:
            变异概率
        """


        self.fitness_func = fitness_func

        self.bounds = bounds

        self.population_size = population_size

        self.generations = generations

        self.mutation_rate = mutation_rate



    def initialize_population(self):

        """
        初始化种群
        """

        population=[]


        for _ in range(self.population_size):

            individual=[]

            for low,high in self.bounds:

                value=np.random.uniform(
                    low,
                    high
                )

                individual.append(value)


            population.append(individual)


        return np.array(population)



    def selection(self,population):

        """
        选择优秀个体
        """

        fitness=np.array(
            [
                self.fitness_func(x)
                for x in population
            ]
        )


        index=np.argsort(
            fitness
        )[-self.population_size//2:]


        return population[index]



    def crossover(self,parent1,parent2):

        """
        交叉
        """

        point=random.randint(
            1,
            len(parent1)-1
        )


        child=np.concatenate(
            (
                parent1[:point],
                parent2[point:]
            )
        )


        return child



    def mutation(self,individual):

        """
        变异
        """

        for i in range(len(individual)):

            if random.random()<self.mutation_rate:

                low,high=self.bounds[i]

                individual[i]=np.random.uniform(
                    low,
                    high
                )


        return individual



    def run(self):

        population=self.initialize_population()


        best=None


        for _ in range(self.generations):


            population=self.selection(
                population
            )


            new_population=[]


            while len(new_population)<self.population_size:


                p1,p2=random.sample(
                    list(population),
                    2
                )


                child=self.crossover(
                    p1,
                    p2
                )


                child=self.mutation(
                    child
                )


                new_population.append(child)



            population=np.array(
                new_population
            )


        scores=[
            self.fitness_func(x)
            for x in population
        ]


        best=population[
            np.argmax(scores)
        ]


        return best