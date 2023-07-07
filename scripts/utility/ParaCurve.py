class ParaCurve:
    "用于处理抛物线拟合得到的中线"

    def __init__(self, PI: float, PJ: float) -> None:
        "初始化"
        self.PI, self.PJ = PI, PJ

    def set(self, a_: float, b_: float, c_: float) -> None:
        "设置抛物线参数"
        self.a, self.b, self.c = a_, b_, c_
        self.da = 4 * self.a * self.a
        self.db = 6 * self.a * self.b
        self.dc = 4 * self.a * (self.c - self.PJ) + 2 * self.b * self.b + 2
        self.dd = 2 * self.b * (self.c - self.PJ) - 2 * self.PI
        self.dda = 12 * self.a * self.a
        self.ddb = 12 * self.a * self.b
        self.ddc = 4 * self.a * (self.c - self.PJ) + 2 * self.b * self.b + 2

    def val(self, x: float) -> float:
        "计算函数值"
        return self.a * x * x + self.b * x + self.c

    def vald(self, x: float) -> float:
        "计算斜率"
        return 2 * self.a * x + self.b

    def perpendicular(self) -> float:
        "用牛顿迭代法求过小车点在抛物线上的垂足"

        def calc(x: float) -> float:
            return self.da * x * x * x + self.db * x * x + self.dc * x + self.dd

        def calcd(x: float) -> float:
            return self.dda * x * x + self.ddb * x + self.ddc

        x = self.PI
        for _ in range(5):
            x -= calc(x) / calcd(x)
        return x


__all__ = ["ParaCurve"]

