"""
参考：
https://blog.csdn.net/u011023470/article/details/111381695
https://blog.csdn.net/u011023470/article/details/111381298
"""
from math import sqrt


class Polyfit2d:
    """
    二次曲线拟合类
    """

    def reset(self) -> None:
        self.n = 0
        self.x = self.y = self.x2 = self.x3 = self.x4 = self.xy = self.x2y = 0

    def update(self, x_: float, y_: float) -> None:
        """
        增加一组数据

        Args:
            x (float): 自变量
            y (float): 因变量
        """
        self.n += 1
        self.x += x_
        self.y += y_
        x2_ = x_ * x_
        self.x2 += x2_
        self.x3 += x2_ * x_
        self.x4 += x2_ * x2_
        self.xy += x_ * y_
        self.x2y += x2_ * y_

    def fit(self) -> None:
        self.x /= self.n
        self.y /= self.n
        self.x2 /= self.n
        self.x3 /= self.n
        self.x4 /= self.n
        self.xy /= self.n
        self.x2y /= self.n
        B = ((self.x * self.y - self.xy) / (self.x3 - self.x2 * self.x) - (self.x2 * self.y - self.x2y) / (self.x4 - self.x2 * self.x2)
             ) / ((self.x3 - self.x2 * self.x) / (self.x4 - self.x2 * self.x2) - (self.x2 - self.x * self.x) / (self.x3 - self.x2 * self.x))
        A = (self.x2y - self.x2 * self.y - (self.x3 - self.x * self.x2)
             * B) / (self.x4 - self.x2 * self.x2)
        C = self.y - self.x2 * A - self.x * B
        self.res = [A, B, C]

    def twoPoints(self, x0, y0, x1, y1):
        """
        以 (x0,y0) 为极值点，获取过 (x1,y1) 的抛物线
        """
        dx = x1 - x0
        if dx == 0:
            A = 0
        else:
            A = (y1 - y0) / (dx * dx)
        B = x0 * (-2) * A
        C = y0 - A * x0 * x0 - B * x0
        self.res = [A, B, C]

    def shift(self, X_POS: int, WIDTH: float, direction: bool) -> None:
        """
        将拟合得到的抛物线延 x0 处的切线的垂线平移一段距离

        Args:
            X_POS (int): 原抛物线上目标点的横坐标
            WIDTH (float): 所要平移的距离
            direction (bool): 平移方向

        Returns:
            List[float]: 新抛物线的3个参数
        """
        A, B, C = self.res
        t = (B - A * X_POS * X_POS - B * X_POS - C) / X_POS
        q = WIDTH / sqrt(t * t + 1)
        if direction:
            q = -q
        p = t * q
        if 2 * A * X_POS + B < 0:
            p = -p
        self.res = [A, B - 2 * A * p, A * p * p - B * p + C + q]

    def val(self, x: float) -> float:
        """
        计算抛物线在 x 处的值
        """
        A, B, C = self.res
        return A * x * x + B * x + C


__all__ = ["Polyfit2d"]
