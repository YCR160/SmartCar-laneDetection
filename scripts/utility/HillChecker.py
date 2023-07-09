from Config import *

# 越来越小


class HillChecker:
    """
    检测坡道的位置
    """

    def __init__(self) -> None:
        self.first = self.last = 0.0  # 坡道的第一个和最后一个点的 y 坐标

    def reset(self):
        self.detected = False   # 是否已经检测到坡道的位置
        self.n = 0              # 坡道的点数

    def update(self, pj: float) -> None:
        """
        将当前点的坐标加入到检测坡道位置的计算中
        param pj: 当前点的 y 坐标
        """
        if self.detected:
            return
        self.n += 1
        if self.n == 1:
            self.first = self.last = pj
        # 如果点数是 4 的倍数，则检查当前点的 y 坐标是否大于上一个点，如果是则认为检测到坡道
        if not self.n & 3:
            if pj > self.last:
                self.detected = True
            else:
                self.last = pj

    def check(self):
        return not self.detected and self.n > HILL_COUNT

    def calc(self) -> None:
        """
        计算坡道的高度
        """
        return self.first - self.last


__all__ = ["HillChecker"]
