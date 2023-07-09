"""
参考：
https://blog.csdn.net/liyuanbhu/article/details/50889951
"""

from math import sqrt


class CircleFit:
    """
    用于拟合一组二维点的圆形
    """

    def __init__(self, circle, text):
        self.circle = circle
        self.text = text

    def reset(self) -> None:
        self.n = 0
        self.x = self.y = self.x2 = self.y2 = self.x3 = self.y3 = self.xy = self.xy2 = self.x2y = 0.0
        self.preX = self.preY = 0.0

    def checkPre(self, x_: float, y_: float) -> bool:
        """
        检查当前点与上一个点之间的距离是否小于 100，如果小于则返回 True，否则返回 False
        """
        if not self.n:
            return True
        dx = x_ - self.preX
        dy = y_ - self.preY
        return dx * dx + dy * dy < 100

    def update(self, x_: float, y_: float) -> None:
        """
        将点加入到拟合圆形的数据集
        """
        if not self.checkPre(x_, y_):
            self.lost()
            return
        self.preX = x_
        self.preY = y_
        self.n += 1
        x2_ = x_ * x_
        y2_ = y_ * y_
        self.x += x_
        self.y += y_
        self.x2 += x2_
        self.y2 += y2_
        self.x3 += x2_ * x_
        self.y3 += y2_ * y_
        self.xy += x_ * y_
        self.xy2 += x_ * y2_
        self.x2y += x2_ * y_

    def fit(self) -> None:
        """
        拟合一组二维点的圆形，计算圆心坐标和半径
        """
        C = self.n * self.x2 - self.x * self.x
        D = self.n * self.xy - self.x * self.y
        E = self.n * self.x3 + self.n * self.xy2 - (self.x2 + self.y2) * self.x
        G = self.n * self.y2 - self.y * self.y
        H = self.n * self.x2y + self.n * self.y3 - (self.x2 + self.y2) * self.y
        a = (H * D - E * G) / (C * G - D * D)
        b = (H * C - E * D) / (D * D - G * C)
        c = -(a * self.x + b * self.y + self.x2 + self.y2) / self.n

        self.X = a / (-2)
        self.Y = b / (-2)
        self.R = sqrt(a * a + b * b - 4 * c) / 2

    def lost(self) -> bool:
        """
        拟合圆形并在图像上绘制圆形和圆心坐标，如果当前点数小于 5，则返回 False
        """
        if self.n < 5:
            if self.n:
                self.reset()
            return False

        self.fit()
        if 18 < self.R < 100:
            self.circle(self.X, self.Y, self.R)
            self.text("(%.1f, %.1f)" % (self.X, self.Y), (self.X, self.Y))
            self.text("%.1f" % self.R, (self.X + 10, self.Y))

        self.reset()
        return True


# 模块的声明语句，该模块中包含的公共接口：CircleFit
# 当其他程序使用 from CircleFit import * 导入该模块时，只有在 __all__ 列表中指定的名称才会被导入
__all__ = ["CircleFit"]
