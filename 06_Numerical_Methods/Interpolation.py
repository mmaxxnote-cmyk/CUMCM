import numpy as np

from scipy.interpolate import interp1d



class Interpolation:


    def __init__(
            self,
            x,
            y
    ):

        """
        x:
            已知横坐标

        y:
            已知纵坐标
        """

        self.x=np.array(x)

        self.y=np.array(y)



    def linear(self):

        """
        线性插值
        """

        return interp1d(
            self.x,
            self.y,
            kind="linear"
        )



    def cubic(self):

        """
        三次样条插值
        """

        return interp1d(
            self.x,
            self.y,
            kind="cubic"
        )