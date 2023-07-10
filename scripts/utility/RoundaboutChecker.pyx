"用于检测环路的状态"
from Config import *

"""
flag:
0.初始
1.上升
2.下降
-1.失效
"""


class CircleHelper:
    """
    检测圆形的位置
    """

    def reset(self) -> None:
        self.prev_x = 0
        self.flag = 0

    def update(self, x: int) -> None:
        # 如果失效，直接返回
        if self.flag == -1:
            return
        # 初始状态，进行更新
        if self.flag == 0:
            self.flag = 1
            self.count = 0
        # 上升状态，出现下降
        elif self.flag == 1 and x < self.prev_x:
            # 如果上升的次数不够，flag 失效
            if self.count < ROUND_UPCOUNT:
                self.flag = -1
            # 否则，进入下降状态
            else:
                self.flag = 2
                self.count = 0
        # 下降状态，出现上升，flag 失效
        elif x > self.prev_x:
            self.flag = -1
        # 更新 prev_x 和 count
        self.prev_x = x
        self.count += 1

    def check(self) -> bool:
        return self.flag == 2 and self.count >= ROUND_DOWNCOUNT


"""
flag:
0.丢线6次
1.圆环
2.丢线一段距离
3.找到3次
-1.失效
"""


class RoundaboutChecker:
    """
    检测车辆是否在环路上行驶
    """

    def __init__(self) -> None:
        self.leftCheck = CircleHelper()     # 左侧圆形检测器
        self.rightCheck = CircleHelper()    # 右侧圆形检测器
        self.flag = 0
        self.count = 0
        self.side = False

    def reset(self):
        self.flag = 0
        self.count = 0

    def lost(self) -> None:
        """
        处理车辆离开环形道路的情况
        """
        # TODO
        if self.flag == -1:
            return

        if self.flag == 0:
            self.count += 1

        elif self.flag == 1:
            if not self.checkCircle():
                self.flag = -1
            else:
                self.flag = 2
                self.count = 1

        elif self.flag == 2:
            self.count += 1

        elif self.count < ROUND_COUNT3:
            self.flag = -1

    def update(self, width: float, pi_: float, l: int, r: int) -> None:
        """
        更新环路状态
        :param width: 环路宽度
        :param pi_: 环路弧度
        :param l: 左侧距离
        :param r: 右侧距离
        :return: None
        """
        # TODO
        if self.flag == -1:
            return
        if width > ROUND_MAXWIDTH:
            return self.lost()
        if self.flag == 0:
            if self.count >= ROUND_COUNT0:
                self.flag = 1
                self.count = 1
                self.leftCheck.reset()
                self.rightCheck.reset()
            else:
                self.flag = -1

        elif self.flag == 1:
            self.pi = pi_
            self.leftCheck.update(l)
            #self.rightCheck.update(r)
        elif self.flag == 2:
            if self.pi - pi_ < ROUND_DIST2:
                self.flag = -1
            else:
                self.flag = 3
                self.count = 1
        else:
            self.count += 1

    def checkCircle(self) -> bool:
        """
        检查左右两侧的圆形检测器是否符合要求
        """
        # TODO
        self.side = self.rightCheck.check()
        return self.leftCheck.check() ^ self.side

    def check(self) -> int:
        # TODO
        return self.flag == 3 and self.count >= ROUND_COUNT3


__all__ = ["RoundaboutChecker"]
