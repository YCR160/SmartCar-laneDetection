"""
参考：
https://blog.csdn.net/liyuanbhu/article/details/50889951
"""

from math import sqrt


class CircleFit:
    def __init__(self, circle, text):
        self.circle = circle
        self.text = text

    def reset(self) -> None:
        self.n = 0
        self.x = self.y = self.x2 = self.y2 = self.x3 = self.y3 = self.xy = self.xy2 = self.x2y = 0.0
        self.preX = self.preY = 0.0

    def checkPre(self, x_: float, y_: float) -> bool:
        if not self.n:
            return True
        dx = x_ - self.preX
        dy = y_ - self.preY
        return dx * dx + dy * dy < 100

    def update(self, x_: float, y_: float) -> None:
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


__all__ = ["CircleFit"]
