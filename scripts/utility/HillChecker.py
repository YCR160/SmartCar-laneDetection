from Config import *

# 越来越小


class HillChecker:
    def __init__(self) -> None:
        self.first = self.last = 0.0

    def reset(self):
        self.isNot = False
        self.n = 0

    def update(self, pj: float) -> None:
        if self.isNot:
            return
        self.n += 1
        if self.n == 1:
            self.first = self.last = pj
        if not self.n & 3:
            if pj > self.last:
                self.isNot = True
            else:
                self.last = pj

    def check(self):
        return not self.isNot and self.n > HILL_COUNT

    def calc(self) -> None:
        return self.first - self.last


__all__ = ["HillChecker"]

