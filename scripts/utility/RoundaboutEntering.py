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
        if abs(pj_ - pj) > ROUND_MAXWIDTH:
            return self.lost()
        if self.flag == 0:
            self.count += 1
        else:
            self.count = self.flag = 1
        self.i, self.j = pi, pj

    def lost(self) -> None:
        if self.flag == 0:
            #print("get", self.count)
            if self.count >= ROUNDENTER_GETCOUNT:
                self.flag = 1
                self.count = 1
            else:
                self.count = 0
        elif self.flag == 1:
            self.count += 1
            #print("lost", self.count)
            if self.count == ROUNDENTER_LOSTCOUNT:
                self.flag = 2

    def check(self) -> bool:
        return self.flag == 2


__all__ = ["RoundaboutEntering"]

