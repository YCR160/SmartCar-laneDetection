from math import sqrt, cos, acos, degrees, radians, atan2
from scripts.transform import axisTransform as _axis
from functools import partial
from Config import *


def vectorCos(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算两个向量之间的余弦值
    param x1, y1: 向量 1 的坐标
    param x2, y2: 向量 2 的坐标
    return: 两个向量之间的余弦值
    """
    return (x1 * x2 + y1 * y2) / sqrt((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算两个点之间距离的平方
    param x1, y1: 点 1 的坐标
    param x2, y2: 点 2 的坐标
    return: 两个点之间距离的平方
    """
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


class FrontForkChecker:
    def __init__(self, perMat, line) -> None:
        self.n = 0          # 三岔路的点数
        self.pi = [0.0] * 4  # 三岔路的四个顶点的 x 坐标
        self.pj = [0.0] * 4  # 三岔路的四个顶点的 y 坐标
        self.axisTransform = partial(
            _axis, perMat=perMat)  # 将三岔路的坐标从图像坐标系转换到车辆坐标系
        self.line = line    # 三岔路的中心线，用于计算三岔路的倾斜角度
        self.res = False    # 三岔路的检测结果

    def reset(self) -> None:
        self.n = 0
        self.res = False

    def lost(self) -> None:
        self.n = 0

    def check(self) -> bool:
        """
        检查三岔路的位置是否符合要求，如果符合则返回 True，否则返回 False
        """
        if self.n < 4:
            return False
        x1 = self.pi[(self.n + 1) & 3] - self.pi[self.n & 3]
        x2 = self.pi[(self.n + 2) & 3] - self.pi[(self.n + 3) & 3]
        if x1 < 0 or x2 < 0:
            return False
        y1 = self.pj[self.n & 3] - self.pj[(self.n + 1) & 3]
        y2 = self.pj[(self.n + 3) & 3] - self.pj[(self.n + 2) & 3]
        res = vectorCos(x1, y1, x2, y2)
        return cos(radians(FORKHIGH)) < res < cos(radians(FORKLOW))

    def calc_abc_from_line_2d(self, x0, y0, x1, y1):
        """
        根据两点坐标计算直线的一般式方程
        param x0, y0: 直线上的第一个点的坐标
        param x1, y1: 直线上的第二个点的坐标
        return: 直线的一般式方程的三个参数
        """
        a = y0 - y1
        b = x1 - x0
        c = x0*y1 - x1*y0
        return a, b, c

    def get_line_cross_point(self, line1, line2):
        """
        计算两条直线的交点坐标
        param line1: 直线 1 的两个点的坐标
        param line2: 直线 2 的两个点的坐标
        return: 两条直线的交点坐标
        """
        # x1y1x2y2
        a0, b0, c0 = self.calc_abc_from_line_2d(*line1)
        a1, b1, c1 = self.calc_abc_from_line_2d(*line2)
        D = a0 * b1 - a1 * b0
        if D == 0:
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        # print(x, y)
        return x, y

    def draw(self):
        """
        在图像上绘制三岔路的位置和交点
        """
        pts = [(self.pi[i & 3], self.pj[i & 3])
               for i in range(self.n, self.n + 4)]
        # line1 = [pts[0][0],pts[0][1],pts[1][0],pts[1][1]]
        # line2 = [pts[2][0],pts[2][1],pts[3][0],pts[3][1]]
        line1 = pts[0] + pts[1]
        line2 = pts[2] + pts[3]
        cross_pt = self.get_line_cross_point(line1, line2)
        print(cross_pt)
        self.Yaw = -(cross_pt[1] - 120)/(cross_pt[0] - 125)
        self.line(pts[0], pts[1], (255, 0, 0), 4)
        self.line(pts[2], pts[3], (255, 0, 0), 4)

    def update(self, i: int, j: int):
        """
        将当前点的坐标加入到检测三岔路位置的计算中
        """
        if self.res:
            return
        if not FORKUPCUT < i < N - FORKDOWNCUT:
            self.lost()
        else:
            self.pi[self.n & 3], self.pj[self.n & 3] = self.axisTransform(i, j)
            self.n += 1
            if self.check():
                self.draw()
                self.res = True


class SideForkChecker:
    def __init__(self, line) -> None:
        self.line = line
        self.n = 0              # 侧三岔路的点数
        self.pi = [0.0] * 7     # 侧三岔路的七个顶点的 x 坐标
        self.pj = [0.0] * 7     # 侧三岔路的七个顶点的 y 坐标
        self.hasLost = False    # 是否已经丢失侧三岔路的位置
        self.res = False        # 侧三岔路的检测结果

    def reset(self) -> None:
        self.n = 0
        self.res = False
        self.hasLost = False

    def lost(self) -> None:
        self.hasLost = True

    def checkDist(self, pi_: float, pj_: float) -> bool:
        if not self.n:
            return True
        return dist2(pi_, pj_, self.pi[(self.n - 1) % 7], self.pj[(self.n - 1) % 7]) < FORKMAXDIST2

    def check(self) -> bool:
        if self.n < 7:
            return False
        r1 = vectorCos(self.pi[(self.n + 1) % 7] - self.pi[(self.n + 2) % 7], self.pj[(self.n + 1) % 7] - self.pj[(self.n + 2) %
                       7], self.pi[(self.n + 5) % 7] - self.pi[(self.n + 4) % 7], self.pj[(self.n + 5) % 7] - self.pj[(self.n + 4) % 7])
        r2 = vectorCos(self.pi[self.n % 7] - self.pi[(self.n + 2) % 7], self.pj[self.n % 7] - self.pj[(self.n + 2) % 7],
                       self.pi[(self.n + 6) % 7] - self.pi[(self.n + 4) % 7], self.pj[(self.n + 6) % 7] - self.pj[(self.n + 4) % 7])
        return cos(radians(FORKHIGH)) < r1 < cos(radians(FORKLOW)) and cos(radians(FORKHIGH)) < r2 < cos(radians(FORKLOW))

    def draw(self) -> None:
        I = (self.n % 7, (self.n + 2) % 7, (self.n + 4) % 7, (self.n + 6) % 7)
        pts = [(self.pi[i], self.pj[i]) for i in I]
        self.line(pts[0], pts[1], (255, 0, 0), 4)
        self.line(pts[2], pts[3], (255, 0, 0), 4)

    def update(self, pi_: float, pj_: float):
        if self.res or self.hasLost:
            return
        if not self.checkDist(pi_, pj_):
            self.lost()
            return
        self.pi[self.n % 7] = pi_
        self.pj[self.n % 7] = pj_
        self.n += 1
        if self.check():
            self.draw()
            self.res = True


__all__ = ["FrontForkChecker", "SideForkChecker"]
