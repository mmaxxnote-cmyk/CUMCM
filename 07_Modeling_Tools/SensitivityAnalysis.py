import numpy as np



class SensitivityAnalysis:


    def __init__(
            self,
            model_func
    ):

        """
        model_func:
            模型函数

            输入参数
            返回结果
        """

        self.model_func=model_func



    def one_factor(
            self,
            params,
            index,
            change=0.1
    ):

        """
        单因素敏感性分析


        params:
            参数列表


        index:
            改变哪个参数


        change:
            变化比例
        """


        params=np.array(
            params,
            dtype=float
        )


        base=self.model_func(
            params
        )


        new_params=params.copy()


        new_params[index]*=(1+change)


        new=self.model_func(
            new_params
        )


        sensitivity=(

            (new-base)/base

        )/change


        return sensitivity



    def range_analysis(
            self,
            params,
            index,
            values
    ):

        """
        参数变化范围分析
        """


        results=[]


        for v in values:

            new_params=np.array(
                params,
                dtype=float
            )


            new_params[index]=v


            results.append(
                self.model_func(
                    new_params
                )
            )


        return results