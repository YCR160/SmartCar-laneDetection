"用于判断是否进入环形道路的类"
from Config import *

"""
flag:
0.搜到5次
1.丢线3次
"""


class RoundaboutEntering:
    def reset(self) -> None:
        self.flag = 0
        self.count = 0
        self.i = self.j = 0.0

    def update(self, pi: float, pj: float, pj_: float) -> None:
        """
        更新环形道路的状态
        :param pi: 直道中线能延伸到的最远的点所在的行
        :param pj: 直道中线能延伸到的最远的点向左寻找到的边线的列，即左上角点
        :param pj_: 直道中线能延伸到的最远的点向右寻找到的边线的列，即右上角点
        :return: None
        """
        # 最大有效宽度 ROUND_MAXWIDTH，大于这个宽度视为丢线
        if abs(pj_ - pj) > ROUND_MAXWIDTH:
            return self.lost()
        if self.flag == 0:
            self.count += 1
        else:
            self.count = self.flag = 1
        self.i, self.j = pi, pj # 这里存储的是进圆环方向的角点

    def lost(self) -> None:
        if self.flag == 0:
            if self.count >= ROUNDENTER_GETCOUNT: # 入环时从上往下搜的最少有效点
                self.flag = 1
                self.count = 1
            else:
                self.count = 0
        elif self.flag == 1:
            self.count += 1
            if self.count == ROUNDENTER_LOSTCOUNT: # 搜到有效点后再至少丢的行数
                self.flag = 2

    def check(self) -> bool:
        return self.flag == 2


__all__ = ["RoundaboutEntering"]

