# cython: language_level=3
import numpy as np
from typing import Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *
from random import randint
from math import atan
from Config import *
from math import sqrt, cos, acos, degrees, radians, atan2
import time
from .ImgShow import ShowImg 
import serial
import struct
import cv2
ser = serial.Serial('/dev/ttyPS1', 115200, timeout=1)
global lastYaw
global j1
global j2, c, rFlag
c = 0
j1 = 0
j2 = 0
lastYaw = 0
rFlag = 0
class PointEliminator:
    "通过判断新的点和前面点的连线斜率是否在特定区间来决定是否保留这个点"

    def __init__(self, main: "ImgProcess") -> None:
        self.main = main
        self.I = [0.0] * 2
        self.J = [0.0] * 2

    def reset(self, invert: bool, fitter: Polyfit2d, color: Tuple[int] = (255, 0, 255)) -> None:
        self.n = 0
        self.invert = invert
        self.fitter = fitter
        self.color = color

    def insert(self, i: float, j: float) -> None:
        self.I[self.n & 1] = i
        self.J[self.n & 1] = j
        self.n += 1

    def check(self, i: float, j: float) -> bool:
        k = (j - self.J[self.n & 1]) / (i - self.I[self.n & 1])
        if self.invert:
            k = -k
        return K_LOW < k < K_HIGH

    def update(self, i: float, j: float) -> None:
        if self.n < 2:
            self.insert(i, j)
        elif self.check(i, j):
            self.insert(i, j)
            self.fitter.update(i, j)
            #self.main.ppoint((i, j), self.color)
        else:
            self.n = 0


class ImgProcess:
    "图像处理类"

    def __init__(self) -> None:
        """图像处理类

        Args:
            Config (dict): 通过 getConfig() 获取的配置
        """
        self.fitter = [Polyfit2d() for u in range(2)]
        self.pointEliminator = [PointEliminator(self) for u in range(2)]
        self.applyConfig()
        self.paraCurve = ParaCurve(self.PI, self.PJ)
        self.hillChecker = [HillChecker() for u in range(2)]
        self.frontForkChecker = FrontForkChecker(self.PERMAT, self.pline)
        self.sideForkChecker = [SideForkChecker(self.pline) for u in range(2)]
        self.roundaboutChecker = RoundaboutChecker()
        self.roundaboutEntering = RoundaboutEntering()

        self.landmark = {"StartLine": False, "Hill": False, "Roundabout1": False, "Fork": False, "Yaw": 0.0}
        #print("ImgProcess")

    def setImg(self, img: np.ndarray) -> None:
        """设置当前需要处理的图像

        Args:
            img (np.ndarray): 使用 cv2.imread(xxx, 0) 读入的灰度图
        """
        self.image_data = img
        self.img = img.tolist()

        # for i in range(N):
        #     for j in range(M):
        #         img[i, j] = 255 if self.isEdge(i, j) else 0
        # self.img = img.tolist()

    def applyConfig(self) -> None:
        "从main窗口获取图像处理所需参数"
        self.PERMAT = getPerMat(SRCARR, PERARR)  # 逆透视变换矩阵
        self.REPMAT = getPerMat(PERARR, SRCARR)  # 反向逆透视变换矩阵

        self.SI, self.SJ = N - 1, M >> 1
        self.PI, self.PJ = axisTransform(self.SI, self.SJ, self.PERMAT)
        self.PI = PI
        #print(f"PI {self.PI}\nPJ {self.PJ}")
        #print(f"FORKLOW {cos(radians(FORKHIGH))}f\nFORKHIGH {cos(radians(FORKLOW))}f")

    def point(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt
        I, J = axisTransform(i, j, self.PERMAT)

    def ppoint(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt
        I, J = axisTransform(i, j, self.REPMAT)

    def line(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        (i1, j1), (i2, j2) = p1, p2
        pi1, pj1 = axisTransform(i1, j1, self.PERMAT)
        pi2, pj2 = axisTransform(i2, j2, self.PERMAT)

    def pline(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        (i1, j1), (i2, j2) = p1, p2
        pi1, pj1 = axisTransform(i1, j1, self.REPMAT)
        pi2, pj2 = axisTransform(i2, j2, self.REPMAT)
    def sobel(self, i: int, j: int, lr: int = LRSTEP) -> int:
        "魔改的sobel算子"
        il = max(CUT, i - UDSTEP)
        ir = min(N - 1, i + UDSTEP)
        jl = max(PADDING, j - lr)
        jr = min(M - PADDING - 1, j + lr)
        #print(self.img[il][jl] , self.img[ir][jr])
        return abs(self.img[il][jl] - self.img[ir][jr]) + abs(self.img[il][j] - self.img[ir][j]) + abs(self.img[i][jl] - self.img[i][jr]) + abs(self.img[il][jr] - self.img[ir][jl])

    def isEdge(self, i: int, j: int):
        "检查(i, j)是否是边界"
        return self.sobel(i, j) >= THRESHLOD

    def checkI(self, i: int) -> bool:
        "检查i是否没有越界"
        return CUT <= i < N

    def checkJ(self, j: int) -> bool:
        "检查j是否没有越界"
        return PADDING <= j < M - PADDING

    def checkCornerIJ(self, i: int, j: int) -> bool:
        "找前沿线时限定范围"
        return self.checkI(i) and j * N > CORNERCUT * (N - i) and N * j < CORNERCUT * i + N * (M - CORNERCUT)
    def calcK(self, i, k):
        "以行号和'斜率'计算列号"
        b = (M >> 1) - (k * (N - 1) // 3)
        return ((k * i) // 3) + b 
    def searchK(self, k: int, draw: bool = False, color: Tuple[int] = None) -> int:
        "沿'斜率'k搜索黑色"
        #if draw and color is None:
        #    color = (randint(0, 255), randint(0, 255), randint(0, 255))a
        i = N - 1
        while True:
            i -= 1
            j = self.calcK(i, k) 
            if not (self.checkCornerIJ(i, j) and self.canny[i, j] == 0):
                return i + 1
    def searchRow(self, i: int, j: int, isRight: bool, draw: bool = False, color: Tuple[int] = None) -> int:
        "按行搜索左右的黑色"
        if isRight:
            s = np.nonzero(self.canny[i,j:M-1])
            if len(s[0]) == 0:
                ans = M - 1
            else:
                ans = s[0][0] + j
        else:
            s = np.nonzero(self.canny[i,0:j])
            if len(s[0]) == 0:
                ans = 0
            else:
                ans = s[0][-1]    
        return ans
    def searchRow2(self, i: int, j: int, isRight: bool, draw: bool = False, color: Tuple[int] = None) -> int:
        "按行搜索左右的黑色"
        if draw and color is None:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
        STEP = 1 if isRight else -1
        while self.checkJ(j) and not self.isEdge(i, j):
            j += STEP
        return j
    def searchRow1(self, i: int, j: int, isRight: bool, last: int) -> int:
        "按行搜索左右的黑色"
        STEP = 1 if isRight else -1
        if isRight and not (last == M - 1):
            j = last - 5
        if not isRight and not (last == 0):
            j = last + 5
        #while self.checkJ(j) and not self.isEdge(i, j):
        while self.canny[i, j] == 0 and self.checkJ(j):   
            j += STEP
        return j
    def getK(self, draw: bool = False) -> None:
        "获取最远前沿所在的'斜率'K"
        self.I = self.K = 0x7FFFFFFF
        self.frontForkChecker.reset()
        i = 0
        for k in range(-6, 7):
            i = self.searchK(k, draw)
            if self.checkCornerIJ(i - 1, self.calcK(i - 1, k)):
                self.frontForkChecker.update(i - 1, self.calcK(i - 1, k))
            else:
                self.frontForkChecker.lost()
            if i < self.I:
                self.I, self.K = i, k

    def getEdge(self, draw: bool = False):
        "逐行获取边界点"
        #self.canny = cv2.Canny(self.image_data, 50, 150)
        #lastside = [0, M - 1]
        J = 0
        width = 0
        self.roundaboutChecker.reset()
        for u in range(2):
            self.fitter[u].reset()
            self.hillChecker[u].reset()
            self.pointEliminator[u].reset(u ^ 1, self.fitter[u], COLORS[u + 4])
            self.sideForkChecker[u].reset()
        #print(N - 1, self.I - 1)
        for I in range(N - 1, self.I - 1, -2):
            J = self.calcK(I, self.K)
            #s1 = np.nonzero(self.canny[I,0:J])
            #s2 = np.nonzero(self.canny[I,J:M-1])
            #if len(s1[0]) == 0 and len(s2[0]) == 0:
            #    side = [0, M - 1]
            #elif len(s1[0]) == 0:
            #    side = [0, s2[0][0]+J]
            #elif len(s2[0]) == 0:
            #    side = [s1[0][-1], M - 1]
            #else:
            #    side = [s1[0][-1], s2[0][0] + J]
            side = [self.searchRow(I, J, u) for u in range(2)]
            #print(side)
            #print(lastside)
            #lastside = side
            #print(lastside)
            pj = [0.0] * 2
            nolost = True
            for u in range(2):
                if self.checkJ(side[u]):
                    pi, pj[u] = axisTransform(I, side[u], self.PERMAT)
                    self.sideForkChecker[u].update(pi, pj[u])
                    self.pointEliminator[u].update(pi, pj[u])
                    if I < HILL_CUT:
                        self.hillChecker[u].update(-pj[u] if u else pj[u])
                else:
                    #print("lost")
                    nolost = False
                    self.sideForkChecker[u].lost()
            if nolost:
                width = pj[1] - pj[0]
                #print(width)
                self.roundaboutChecker.update(width, pi, side[0], -side[1])
            else:
                self.roundaboutChecker.lost()
            #print(pi, pj)

    def getMid(self, drawEdge: bool = False) -> bool:
        "获取中线"
        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        #print(px)
        for u in range(2):
            if self.fitter[u].n > 5:
                self.fitter[u].fit()
                self.fitter[u].shift(X_POS, WIDTH, u)

        if min(self.fitter[u].n for u in range(2)) > 5:
            N = sum(self.fitter[u].n for u in range(2))
            a, b, c = [sum(self.fitter[u].res[i] * self.fitter[u].n for u in range(2)) / N for i in range(3)]
        elif max(self.fitter[u].n for u in range(2)) > 5:
            a, b, c = self.fitter[0].res if self.fitter[0].n > 5 else self.fitter[1].res
            #print(a,b,c)
        else:
            return False

        self.paraCurve.set(a, b, c)
        py = [self.paraCurve.val(v) for v in px]
        return True

    def getTarget(self):
        "获取参考点位置"
        x = self.paraCurve.perpendicular()
        y = self.paraCurve.val(x)
        #self.ppoint((round(x), round(y)), (0, 0, 255))

        l, r = x - DIST, x
        #print(x, y)
        for _ in range(5):
            self.X1 = (l + r) / 2
            self.Y1 = self.paraCurve.val(self.X1)
            dx = x - self.X1
            dy = y - self.Y1
            d = dx * dx + dy * dy
            if d < DIST * DIST:
                r = self.X1
            else:
                l = self.X1
        #self.ppoint((round(self.X1), round(self.Y1)), (255, 127, 255), 6)

    def solve(self):
        "获取目标偏航角"
        self.landmark["Yaw"] = atan2(self.Y1 - self.PJ, self.PI - X0 - self.X1)

    def checkStartLine(self, i: int) -> bool:
        "检测起跑线"
        pre = self.sobel(i, STARTLINE_PADDING, 1) > THRESHLOD
        count = 0
        for j in range(STARTLINE_PADDING + 1, M - STARTLINE_PADDING):
            cur = self.sobel(i, j, 1) > THRESHLOD
            count += pre ^ cur
            pre = cur
        return count > STARTLINE_COUNT

    def roundaboutGetCorner(self, U: bool = False) -> bool:
        "入环岛获取上角点"
        self.roundaboutEntering.reset()

        for i in range(self.I + 1, N, 2):
            m = self.calcK(i, self.K)
            side = [self.searchRow(i, m, u) for u in range(2)]
            if self.checkJ(side[U]):
                pi, pj_ = axisTransform(i, side[U ^ 1], self.PERMAT)
                pi, pj = axisTransform(i, side[U], self.PERMAT)
                self.roundaboutEntering.update(pi, pj, pj_)
            else:
                self.roundaboutEntering.lost()
            if self.roundaboutEntering.check():
                return True
        return False

    def roundaboutGetInMid(self, U: bool = False):
        "入环岛获取中线"
        U ^= 1
        dpi, dpj = axisTransform(N - 1, self.searchRow(N - 1, M >> 1, U), self.PERMAT)
        self.ppoint((self.roundaboutEntering.i, self.roundaboutEntering.j)), self.ppoint((dpi, dpj))
        self.fitter[U].twoPoints(dpi, dpj, self.roundaboutEntering.i, self.roundaboutEntering.j)

        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        py = [self.fitter[U].val(v) for v in px]

        self.fitter[U].shift(X_POS, WIDTH, U)
        self.paraCurve.set(*self.fitter[U].res)

        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        py = [self.paraCurve.val(v) for v in px]
        #print(dpi,dpj,px,py)
    def roundaboutGetOutMid(self, U: bool = False):
        U ^= 1

        pi0, pj0 = axisTransform(N - 1, self.searchRow(N - 1, M >> 1, U), self.PERMAT)
        pi1, pj1 = axisTransform(self.I, PADDING if U else M - PADDING, self.PERMAT)
        self.fitter[U].twoPoints(pi0, pj0, pi1, pj1)

        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        py = [self.fitter[U].val(v) for v in px]

        self.fitter[U].shift(X_POS, WIDTH, U)
        self.paraCurve.set(*self.fitter[U].res)

        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        py = [self.paraCurve.val(v) for v in px]


        # 环岛

    def work(self):
        self.canny = cv2.Canny(self.image_data, 250, 350)
        kernel = np.ones((2, 2), np.uint8)
        self.canny = cv2.dilate(self.canny, kernel)
        #cv2.imshow('1',self.canny)
        #cv2.waitKey(1)
        global lastYaw, rFlag
        self.canny2 = cv2.cvtColor(self.canny, cv2.COLOR_GRAY2BGR)
        #print("work")
        "图像处理的完整工作流程"
        self.landmark["StartLine"] = self.checkStartLine(STARTLINE_I1) or self.checkStartLine(STARTLINE_I2)
        self.getK(True)
        #print(self.I)
        #print(self.K)
        "入环"
        #isRight = True
        #self.roundaboutGetCorner(isRight)
        #if self.roundaboutGetCorner(isRight):
            #print("Yes")
        #    self.roundaboutGetInMid(isRight)
        #    self.getTarget()
        #    self.solve()
        "出环"
        #isRight = True
        #self.roundaboutGetOutMid(isRight)
        #self.getTarget()
        #self.solve()
        #print(self.landmark["Yaw"])
        "正常"
        self.getEdge()
        self.landmark["Hill"] = self.hillChecker[0].check() and self.hillChecker[1].check() and self.hillChecker[0].calc() + self.hillChecker[1].calc() > HILL_DIFF
        self.landmark["Roundabout1"] = "None" if not self.roundaboutChecker.check() else "Right" if self.roundaboutChecker.side else "Left"
        self.landmark["Fork"] = self.frontForkChecker.res and (self.sideForkChecker[0].res or self.sideForkChecker[1].res)
        if self.landmark["Roundabout1"] == "Right":
            isRight = True
            if self.roundaboutGetCorner(isRight) and rFlag == 0:
                rFlag = 1
        if rFlag > 0:
            isRight = True
            #self.roundaboutGetCorner(isRight)
            if self.roundaboutGetCorner(isRight):
            #print("Yes")
                if rFlag == 1:
                    rFlag = 2
                self.roundaboutGetInMid(isRight)
                self.getTarget()
                self.solve()
                if rFlag == 3 and self.I < 10:
                    rFlag = 0
            elif rFlag == 2 or rFlag == 3:
                if rFlag == 2:
                    rFlag = 3
                isRight = True
                self.roundaboutGetOutMid(isRight)
                self.getTarget()
                self.solve()
        #print(self.landmark["Yaw"])
        elif self.getMid(True):
        #    #print(1)
            self.getTarget()
            self.solve()
            lastYaw = self.landmark["Yaw"]
        elif self.K >= 0:
            isRight = False
            self.roundaboutGetOutMid(isRight)
            self.getTarget()
            self.solve()
        elif self.K <= 0:
            isRight = True
            self.roundaboutGetOutMid(isRight)
            self.getTarget()
            self.solve()
        print("rFlag=",rFlag)
        if ser.isOpen():
            #print('send')
            str = struct.pack('f',self.landmark["Yaw"])
            ser.write('aa'.encode('utf-8'))
            ser.write(str)
            ser.write('bb'.encode('utf-8'))
        print(self.landmark["Roundabout1"])
        print(self.landmark["Yaw"])
        #SRC = np.array(SRCARR,dtype=np.float32)
        #SRC = SRC[:, [1 , 0]]
        #PER = np.array(PERARR,dtype=np.float32)
        #PER = PER[:, [1 , 0]]
        #H = cv2.getPerspectiveTransform(SRC, PER)
        #WarpedImg = cv2.warpPerspective(self.image_data, H, (200,200))
        #ShowImg(self.image_data, H)
        #cv2.imshow('1',self.canny2)
        #cv2.waitKey(1)
__all__ = ["ImgProcess"]
#
