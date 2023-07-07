from Config import *

"""
flag:
0.初始
1.上升
2.下降
-1.失效
"""


class CircleHelper:
    def reset(self) -> None:
        self.pre = 0
        self.flag = 0

    def update(self, x: int) -> None:
        if self.flag == -1:
            return
        if self.flag == 0:
            self.flag = 1
            self.count = 0
        elif self.flag == 1:
            #print("x=",x)
            #print("self.pre=",self.pre)
            if x < self.pre:
                print("up", self.count)
                if self.count < ROUND_UPCOUNT:
                    self.flag = -1
                else:
                    self.flag = 2
                    self.count = 0
        else:
            if x > self.pre:
                self.flag = -1
        self.pre = x
        self.count += 1
        #print("checkc", self.count)
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
    def __init__(self) -> None:
        self.leftCheck = CircleHelper()
        self.rightCheck = CircleHelper()
        self.flag = 0
        self.count = 0
        self.side = False

    def reset(self):
        self.flag = 0
        self.count = 0

    def lost(self) -> None:
        if self.flag == -1:
            return

        if self.flag == 0:
            self.count += 1

        elif self.flag == 1:
            if not self.checkCircle():
                #print("NO")
                self.flag = -1
            else:
                self.flag = 2
                self.count = 1

        elif self.flag == 2:
            self.count += 1

        else:
            print("count3", self.count)
            if self.count < ROUND_COUNT3:
                self.flag = -1
    def update(self, width: float, pi_: float, l: int, r: int) -> None:
        if self.flag == -1:
            return
        #print("w",width)
        if width > ROUND_MAXWIDTH:
            #print('lostR')
            return self.lost()
        #print("count=",self.count)
        if self.flag == 0:
            print("count0",self.count)
            if self.count >= ROUND_COUNT0:
                self.flag = 1
                self.count = 1
                self.leftCheck.reset()
                self.rightCheck.reset()
            else:
                self.flag = -1

        elif self.flag == 1:
            self.pi = pi_
            #self.leftCheck.update(l)
            self.rightCheck.update(r)
        elif self.flag == 2:
            print("dist2",self.pi - pi_)
            if self.pi - pi_ < ROUND_DIST2:
                self.flag = -1
            else:
                self.flag = 3
                self.count = 1
        else:
            self.count += 1
    def checkCircle(self) -> bool:
        self.side = self.rightCheck.check()
        return self.leftCheck.check() ^ self.side

    def check(self) -> int:
        
        return self.flag == 3 and self.count >= ROUND_COUNT3


__all__ = ["RoundaboutChecker"]

