import pandas as pd



class ModelComparison:


    def __init__(self):

        self.results=[]



    def add_model(
            self,
            name,
            metrics
    ):

        """
        添加模型结果

        name:
            模型名称

        metrics:
            指标字典
        """


        result={
            "Model":name
        }


        result.update(
            metrics
        )


        self.results.append(
            result
        )



    def dataframe(self):

        """
        转换为表格
        """

        return pd.DataFrame(
            self.results
        )



    def rank(
            self,
            metric,
            ascending=True
    ):

        """
        排名

        ascending:
            True表示越小越好
        """


        df=self.dataframe()


        return df.sort_values(
            by=metric,
            ascending=ascending
        )