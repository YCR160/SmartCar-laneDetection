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

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc,60.0,(200,200))
ser = serial.Serial('/dev/ttyPS1', 460800, timeout=1)
global roundC, lastYaw, turnFlag, forkFlag, fS, forkC, wFlag, wS, wC, lineF1, lineF2, lineF3, lineC, redC
global j1, x_last, p_last
global j2, c, rFlag, sFlag
redC = 0
roundC = 0
lineF1 = 0
lineF2 = 0
lineF3 = 0
lineC = 0
wC = 0
wS = 0
wFlag = wF
forkFlag = fF
forkC = 0
fS = 0
turnFlag = 0
x_last = 0
p_last = 0
sFlag = 0
c = 0
j1 = 0
j2 = 0
lastYaw = 0
rFlag = 0


class PointEliminator:
    "通过判断新的点和前面点的连线斜率是否在特定区间来决定是否保留这个点"

    def __init__(self, main: "ImgProcess") -> None:
        self.main = main
        self.I = [0.0] * 2 # 面前直道中线能够延伸的最远距离
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
        """
        用于判断新的点和前面点的连线斜率是否在特定区间，可以去除不必要的点
        Args:
            i (float): 新的点的i坐标
            j (float): 新的点的j坐标
        """
        k = (j - self.J[self.n & 1]) / (i - self.I[self.n & 1])
        if self.invert:
            k = -k
        return K_LOW < k < K_HIGH

    def update(self, i: float, j: float) -> None:
        """
        用于实现对车道线的实时优化和跟踪，跟踪车道线的变化
        Args:
            i (float): 新的点的i坐标
            j (float): 新的点的j坐标
        """
        # 类中点的数量小于 2
        if self.n < 2:
            self.insert(i, j)
        # 判断新的点和前面点的连线斜率在特定区间
        elif self.check(i, j):
            self.insert(i, j)
            self.fitter.update(i, j) # 对车道线进行拟合和优化
            # self.main.ppoint((i, j), self.color)
        else:
            self.n = 0 # 类中点的数量重置为 0


class ImgProcess:
    def __init__(self) -> None:
        # 将一些工具类实例化
        self.fitter = [Polyfit2d() for u in range(2)]
        self.pointEliminator = [PointEliminator(self) for u in range(2)]
        self.applyConfig()
        self.paraCurve = ParaCurve(self.PI, self.PJ)
        self.hillChecker = [HillChecker() for u in range(2)]
        self.frontForkChecker = FrontForkChecker(self.PERMAT, self.pline)
        self.sideForkChecker = [SideForkChecker(self.pline) for u in range(2)]
        self.roundaboutChecker = RoundaboutChecker()
        self.roundaboutEntering = RoundaboutEntering()

        # F 是画面中弯道位于的行号，行号越大，表示弯道距离越近
        self.F = 0

        # 字面意思，地面的标记
        # 用于判断是否到达了某个地标
        # StartLine: 起跑线
        # Hill: 上坡
        # Roundabout: 第一个环岛
        # Fork: 三岔路口
        # Yaw: 车头偏航角
        self.landmark = {
            "StartLine": False,
            "Hill": False,
            "Roundabout": False,
            "Fork": False,
            "Yaw": 0.0}

    def setImg(self, img: np.ndarray) -> None:
        """
        设置当前需要处理的图像

        Args:
            img (np.ndarray): 使用 cv2.imread(xxx, 0) 读入的灰度图
        """
        self.image_data = img   # 当前需要处理的图像
        self.img = img.tolist() # 将 numpy 数组转换为 Python 列表

        # for i in range(N):
        #     for j in range(M):
        #         img[i, j] = 255 if self.isEdge(i, j) else 0
        # self.img = img.tolist()
        # self.SrcShow = ZoomedImg(img, SRCZOOM)
        # self.PerShow = ZoomedImg(transfomImg(img, self.PERMAT, N, M, N_, M_, I_SHIFT, J_SHIFT), PERZOOM)

    def applyConfig(self) -> None:
        "从 main 窗口获取图像处理所需参数"
        self.PERMAT = getPerMat(SRCARR, PERARR)  # 逆透视变换矩阵
        self.REPMAT = getPerMat(PERARR, SRCARR)  # 反向逆透视变换矩阵

        self.SI, self.SJ = N - 1, M >> 1        # 原图中心点坐标，高度为 N，宽度为 M
        self.PI, self.PJ = axisTransform(self.SI, self.SJ, self.PERMAT) # 使用变换矩阵映射坐标
        self.PI = PI                            # 逆透视变换后的图像中心点坐标，定义位于 Config.py

    def point(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt                               # 原图上的坐标
        I, J = axisTransform(i, j, self.PERMAT) # 逆透视变换矩阵

    def ppoint(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt                               # 原图上的坐标
        I, J = axisTransform(i, j, self.REPMAT) # 反向逆透视变换矩阵

    def line(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        "图像上绘制线段"
        (i1, j1), (i2, j2) = p1, p2                     # 原图上的坐标
        pi1, pj1 = axisTransform(i1, j1, self.PERMAT)   # 逆透视变换矩阵
        pi2, pj2 = axisTransform(i2, j2, self.PERMAT)   # 逆透视变换矩阵

    def pline(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        "图像上绘制线段"
        (i1, j1), (i2, j2) = p1, p2                     # 原图上的坐标
        pi1, pj1 = axisTransform(i1, j1, self.REPMAT)   # 反向逆透视变换矩阵
        pi2, pj2 = axisTransform(i2, j2, self.REPMAT)   # 反向逆透视变换矩阵

    def sobel(self, i: int, j: int, lr: int = LRSTEP) -> int:
        """
        魔改的 sobel 算子
        Sobel 算子是一种图像边缘检测算法
        通过计算像素点周围像素点的灰度值差异，来确定像素点是否处于边缘位置
        """
        il = max(CUT, i - UDSTEP)
        ir = min(N - 1, i + UDSTEP)
        jl = max(PADDING, j - lr)
        jr = min(M - PADDING - 1, j + lr)
        return abs(self.img[il][jl] - self.img[ir][jr]) + abs(self.img[il][j] - self.img[ir][j]) + abs(self.img[i][jl] - self.img[i][jr]) + abs(self.img[il][jr] - self.img[ir][jl])

    def isEdge(self, i: int, j: int):
        "检查 (i, j) 是否是边界"
        return self.sobel(i, j) >= THRESHLOD # THRESHLOD 为 sobel的阈值

    def checkI(self, i: int) -> bool:
        "检查 i 是否没有越界"
        return CUT <= i < N # CUT 为裁剪最上面的多少行

    def checkJ(self, j: int) -> bool:
        "检查 j 是否没有越界"
        return PADDING <= j < M - PADDING # PADDING 为舍弃左右边界的大小

    def checkCornerIJ(self, i: int, j: int) -> bool:
        "找前沿线时限定范围"
        return self.checkI(i) and j * N > CORNERCUT * (N - i) and N * j < CORNERCUT * i + N * (M - CORNERCUT) # CORNERCUT 为搜索前沿时舍弃最上面角的宽度

    def calcK(self, i, k):
        "以行号和'斜率'计算列号"
        # cdef int b = (M >> 1) - (k * (N - 1) // 3)
        b = (M >> 1) - (k * (N - 1) // 3)
        return ((k * i) // 3) + b

    def searchK(self, k: int) -> int:
        "沿'斜率'k 搜索黑色，并返回最后一个搜索到的黑色像素点的行号"
        # if draw and color is None:
        #     color = (randint(0, 255), randint(0, 255), randint(0, 255))a
        # cdef int i = N - 1
        # cdef int j = 0
        # cdef int[:,:] canny = np.array(self.canny, dtype = np.int32)
        i = N - 1
        j = 0
        canny = np.array(self.canny, dtype = np.int32)
        while True:
            i -= searchKn
            j = self.calcK(i, k)
            self.canny2[i, j, 0] = 255
            if not (self.checkCornerIJ(i, j) and  canny[i,j + 1] == 0 and canny[i, j] == 0 and canny[i, j - 1] == 0):
                return i + 1

    def searchRow(self, i: int, j: int, isRight: bool, draw: bool = False, color: Tuple[int] = None) -> int:
        """
        在给定的行号i和列号j处，沿着该行向左或向右搜索第一个非零像素点的列号

        Args:
            i: 行号
            j: 列号
            isRight: 是否向右搜索
            draw:
            color:
        """
        # cdef int col = 0 # 用于存储搜索到的列号
        col = 0 # 用于存储搜索到的列号
        if isRight: # 搜索方向是否向右？可能吧……
            # 根据 isRight 的值，计算出需要搜索的像素点范围
            # 找到该范围内第一个非零像素点的位置
            s = np.nonzero(self.canny[i,j:M-1])
            # 如果该范围内没有非零像素点，则返回 M-1，表示搜索到了图像的边缘
            if len(s[0]) == 0:
                col = M - 1
            else:
                col = s[0][0] + j
        else:
            # 根据 isRight 的值，计算出需要搜索的像素点范围
            # 找到该范围内第一个非零像素点的位置
            s = np.nonzero(self.canny[i,0:j])
            # 如果该范围内没有非零像素点，则返回 0，表示搜索到了图像的边缘
            if len(s[0]) == 0:
                col = 0
            else:
                col = s[0][-1]
        return col

    def getK(self, draw: bool = False) -> None:
        "获取最远前沿所在的'斜率'K"
        self.I = self.K = 0x7FFFFFFF
        self.F = 0
        self.frontForkChecker.reset()
        # cdef int i = 0
        i = 0
        for k in range(-6, 7):
            i = self.searchK(k)
            if lastYaw >= 0:
                if abs(k) <= numberCheck and k >= 0: # numberCheck 为查找的斜率数量
                    if k == 0:
                        self.F = (i + self.F)
                    else:
                        self.F = (i * abs(k) * numberF + self.F)
            else:
                if abs(k) <= numberCheck and k <= 0:
                    if k == 0:
                        self.F = (i + self.F)
                    else:
                        self.F = (i * abs(k) * numberF + self.F)
            if self.checkCornerIJ(i - 1, self.calcK(i - 1, k)):
                self.frontForkChecker.update(i - 1, self.calcK(i - 1, k))
            else:
                self.frontForkChecker.lost()
            if i < self.I:
                self.I, self.K = i, k

    def getEdge(self, draw: bool = False):
        "逐行获取边界点"
        self.checkLeft = 0
        # self.canny = cv2.Canny(self.image_data, 50, 150)
        lastside = [0, M - 1]
        left = []
        leftx = []
        right = []
        rightx = []
        # cdef int J = 0
        # cdef float width = 0
        # cdef int[2] side
        # cdef float pi
        # cdef float[2] pj
        J = 0
        width = 0
        side = [0, M - 1]
        pi = 0
        pj = [0, M - 1]
        self.roundaboutChecker.reset()
        for u in range(2):
            self.fitter[u].reset()
            self.hillChecker[u].reset()
            self.pointEliminator[u].reset(u ^ 1, self.fitter[u], COLORS[u + 4])
            self.sideForkChecker[u].reset()
        tot = 0
        for I in range(N - 1, self.I - 1, -2):
            J = self.calcK(I, self.K)
            side = [self.searchRow(I, J, u) for u in range(2)]
            if lastside[0] - side[0] > 10 and self.checkLeft == 0:
                self.checkLeft = I
            lastside = side
            if not side[0] == 0 or not side[1] == M - 1:
                tot = tot + 1
            if I == N - 1:
                self.left = side[0]
                self.right = side[1]
            if not side[0] == 0 and I > FitC:
                left.append(side[0])
                leftx.append(I)
            if not side[1] == M - 1 and I > FitC:
                right.append(side[1])
                rightx.append(I)
            self.canny2[I, side[0], 1] = 100
            self.canny2[I, side[1], 2] = 100
            pj = [0.0] * 2
            nolost = True
            for u in range(2):
                if self.checkJ(side[u]):
                    pi, pj[u] = axisTransform(I, side[u], self.PERMAT)
                    self.sideForkChecker[u].update(pi, pj[u])
                    if I > length or (I <= length and tot < totC):
                        self.pointEliminator[u].update(pi, pj[u])
                    if I < HILL_CUT:
                        self.hillChecker[u].update(-pj[u] if u else pj[u])
                else:
                    nolost = False
                    self.sideForkChecker[u].lost()
            if nolost:
                width = pj[1] - pj[0]
                self.roundaboutChecker.update(width, pi, side[0], -side[1])
            else:
                self.roundaboutChecker.lost()
        reg1 = []
        reg2 = []
        k1 = []
        k2 = []
        if len(left) >= Scheck2:
            k1,reg1,_,_,_=np.polyfit(leftx,left,1,full = True)
        if len(reg1) == 0:
            k1 = [10]
            reg1 = [Scheck * 2]
        if len(right) >= Scheck2:
            k2,reg2,_,_,_=np.polyfit(rightx,right,1,full = True)
        if len(reg2) == 0:
            k2 = [10]
            reg2 = [Scheck * 2]
        self.S = (reg1[0] + reg2[0]) / 2
        self.leftK = abs(k1[0])
        self.rightK = abs(k2[0])

    def getMid(self, drawEdge: bool = False) -> bool:
        "获取中线"
        px = list(range(-I_SHIFT, N_ - I_SHIFT))
        self.rightYaw = 0
        # cdef float a, b, c
        a, b, c = 0, 0, 0
        for u in range(2):
            if self.fitter[u].n > getM:
                self.fitter[u].fit()
                if u == 1:
                    py = [self.fitter[u].val(v) for v in px]
                    k,reg,_,_,_=np.polyfit(px,py,1,full = True)
                    self.rightYaw = -k[0]
                self.fitter[u].shift(X_POS, WIDTH, u)
        if min(self.fitter[u].n for u in range(2)) > getM:
            N = sum(self.fitter[u].n for u in range(2))
            a, b, c = [sum(self.fitter[u].res[i] * self.fitter[u].n for u in range(2)) / N for i in range(3)]
        elif max(self.fitter[u].n for u in range(2)) > getM:
            a, b, c = self.fitter[0].res if self.fitter[0].n > getM else self.fitter[1].res
        else:
            return False

        self.paraCurve.set(a, b, c)
        py = [self.paraCurve.val(v) for v in px]
        return True

    def getTarget(self):
        "获取参考点位置"
        # cdef float x = self.paraCurve.perpendicular()
        # cdef float y = self.paraCurve.val(x)

        # cdef float l = x - DIST
        # cdef float r = x
        # cdef float dx, dy, d
        x = self.paraCurve.perpendicular()  # 垂足
        y = self.paraCurve.val(x)           # 垂足纵坐标

        l = x - DIST    # 垂足向上 DIST
        r = x           # 垂足
        dx, dy, d = 0, 0, 0
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

    def solve(self):
        "获取目标偏航角"
        self.landmark["Yaw"] = atan2(self.Y1 - self.PJ, self.PI - X0 - self.X1)

    def checkStartLine(self, i: int) -> bool:
        "检测起跑线"
        # cdef int pre = self.sobel(i, STARTLINE_PADDING, 1) > THRESHLOD
        # cdef int count = 0
        # cdef int cur
        pre = self.sobel(i, STARTLINE_PADDING, 1) > THRESHLOD
        count = 0
        cur = 0
        for j in range(STARTLINE_PADDING + 1, M - STARTLINE_PADDING):
            cur = self.sobel(i, j, 1) > THRESHLOD
            count += pre ^ cur
            pre = cur
        return count > STARTLINE_COUNT

    def roundaboutGetCorner(self, U: bool = False) -> bool:
        "入环岛获取上角点"
        anss = 0
        self.C = 0
        self.roundaboutEntering.reset()
        # cdef int m
        # cdef float pi, pj_, pj
        m = 0
        pi, pj_, pj = 0, 0, 0
        for i in range(self.I + 1, N, 2): # 从前面直道中线能够延伸的最远距离到画面的下边界
            m = self.calcK(i, self.K) # 以行号和斜率计算列号
            side = [self.searchRow(i, m, u) for u in range(2)] # 从给定的行号 I 和列号 J 处，沿着该行向左或向右搜索第一个非零像素点的列号
            if self.checkJ(side[U]): # 如果没有越界
                pi, pj_ = axisTransform(i, side[U ^ 1], self.PERMAT) # 将行列号转换为坐标，即为右上角点
                pi, pj = axisTransform(i, side[U], self.PERMAT) # 将行列号转换为坐标，即为左上角点
                self.roundaboutEntering.update(pi, pj, pj_)
            else:
                self.roundaboutEntering.lost()
            if self.roundaboutEntering.check():
                if (i < 30):
                    anss = i
                self.roundaboutEntering.reset()
        if anss == 0:
            return False
        else:
            self.C = anss
            return True

    def roundaboutGetInMid(self, U: bool = False):
        "入环岛获取中线"
        U ^= 1
        dpi, dpj = axisTransform(N - 1, self.searchRow(N - 1, M >> 1, U), self.PERMAT) # 将行列号转换为坐标，即为画面最后一行中间的点
        # self.ppoint((self.roundaboutEntering.i, self.roundaboutEntering.j)), self.ppoint((dpi, dpj))
        self.fitter[U].twoPoints(dpi, dpj, self.roundaboutEntering.i, self.roundaboutEntering.j) # 以车所在的点作为极值点，过左上角点拟合抛物线

        # px = list(range(-I_SHIFT, N_ - I_SHIFT))
        # py = [self.fitter[U].val(v) for v in px]

        self.fitter[U].shift(X_POS, WIDTH, U)   # 将拟合的抛物线向左移动 X_POS 个像素
        self.paraCurve.set(*self.fitter[U].res) # 将拟合的抛物线的系数保存到 paraCurve 中

        # px = list(range(-I_SHIFT, N_ - I_SHIFT))
        # py = [self.paraCurve.val(v) for v in px]

    def roundaboutGetOutMid(self, U: bool = False):
        U ^= 1
        pi0, pj0 = axisTransform(N - 1, self.searchRow(N - 1, M >> 1, U), self.PERMAT) # 将行列号转换为坐标，即为画面最后一行中间的点
        pi1, pj1 = axisTransform(self.I, PADDING if U else M - PADDING, self.PERMAT)
        self.fitter[U].twoPoints(pi0, pj0, pi1, pj1)

        # px = list(range(-I_SHIFT, N_ - I_SHIFT))
        # py = [self.fitter[U].val(v) for v in px]

        self.fitter[U].shift(X_POS, WIDTH, U)
        self.paraCurve.set(*self.fitter[U].res)

        # px = list(range(-I_SHIFT, N_ - I_SHIFT))
        # py = [self.paraCurve.val(v) for v in px]

    def kalman(self,z_measure,x_last=0,p_last=0,Q=0.018,R=0.0542):
        x_mid = x_last
        p_mid = p_last + Q
        kg = p_mid/(p_mid + R)
        x_now = x_mid + kg*(z_measure - x_mid)
        p_now = (1-kg)*p_mid
        p_last = p_now
        x_last = x_now
        return x_now,p_last,x_last
    
    def work(self, fcolor1, fcolor2, countRed, countRed2):
        self.rightYaw = 0
        self.C = 0
        # start = time.time()
        # self.image_data = cv2.GaussianBlur(self.image_data,(gauss,gauss),0)
        self.canny = cv2.Canny(self.image_data, cannydown, cannyup)
        kernel = np.ones((kn, kn), np.uint8)
        self.canny = cv2.dilate(self.canny, kernel)
        # ret, self.thresh = cv2.threshold(self.image_data[:,80:175], blackN, 255, cv2.THRESH_BINARY_INV)
        # nonzero = np.nonzero(self.thresh)
        # x = nonzero[0]
        # y = nonzero[1]#mask = cv2.inRange(hsv,lower_red,upper_red)+cv2.inRange(hsv,lower_red2,upper_red2) #提取所需颜色
        # if (len(x)) > 10:
        #     k,reg,_,_,_=np.polyfit(x,y,1,full = True)
        #     self.kBlack = abs(k[0])
        # else:
        #     self.kBlack = 0
        # cv2.imshow('1',self.canny)
        # cv2.waitKey(1)
        global roundC, lastYaw, rFlag, sFlag, p_last, x_last, turnFlag, forkFlag, fS, forkC, wFlag, wS, wC, lineF1, lineF2, lineF3, lineC, redC
        self.Yaw = lastYaw
        self.canny2 = cv2.cvtColor(self.canny, cv2.COLOR_GRAY2BGR)

        "图像处理的完整工作流程"
        self.landmark["StartLine"] = self.checkStartLine(STARTLINE_I1) or self.checkStartLine(STARTLINE_I2)
        self.getK(True)
        "正常"
        self.getEdge()
        self.landmark["Hill"] = self.hillChecker[0].check() and self.hillChecker[1].check() and self.hillChecker[0].calc() + self.hillChecker[1].calc() > HILL_DIFF
        # 如果 side，则 Right
        # 如果 !side，则 Left
        # 如果 check 不返回，则 None
        self.landmark["Roundabout"] = "None" if not self.roundaboutChecker.check() else "Left" if self.roundaboutChecker.side else "Right"
        self.landmark["Fork"] = self.frontForkChecker.res and (self.sideForkChecker[0].res or self.sideForkChecker[1].res)
        midYaw = 0

        # 提前回正
        Kflag = 0
        if self.K <= 0:
            if self.leftK <= Kcheck:
                Kflag = 1
        else:
            if self.rightK <= Kcheck:
                Kflag = 1
        self.Kflag = Kflag
        
        if self.landmark["Roundabout"] == ROUND_DIR or rFlag > 0:
            isRight = False
            if rFlag == 0:
                rFlag = 1
            if self.roundaboutGetCorner(isRight):
                if rFlag == 1:
                    rFlag = 2
                self.roundaboutGetInMid(isRight)
                self.getTarget()
                self.solve()
                if rFlag == 2:
                    self.landmark["Yaw"] = inRoundYaw
                if rFlag == 3 and self.I < outF1 and roundC > roundCheck:
                    rFlag = 0
                    roundC = 0
            elif rFlag == 2 or rFlag == 3:
                roundC = roundC + 1
                if rFlag == 2 and roundC >= roundCheck0:
                    rFlag = 3
                if rFlag == 3 and self.I < outF1 and roundC > roundCheck:
                    rFlag = 0
                    roundC = 0
                isRight = False
                self.roundaboutGetOutMid(isRight)
                self.getTarget()
                self.solve()
                if rFlag == 2 and roundC < roundCheck0:
                    self.landmark["Yaw"] = inRoundYaw

        elif self.getMid(True) and (self.F < Fcheck1 or Kflag == 1):
            self.getTarget()
            self.solve()
        elif self.K >= 0:
            if self.getMid(True):
                self.getTarget()
                self.solve()
                midYaw = self.landmark["Yaw"]
            isRight = False
            self.roundaboutGetOutMid(isRight)
            self.getTarget()
            self.solve()
            if midYaw * self.landmark["Yaw"] < 0:
                self.landmark["Yaw"] = midYaw
        elif self.K <= 0:
            if self.getMid(True):
                self.getTarget()
                self.solve()
                midYaw = self.landmark["Yaw"]
            isRight = True
            self.roundaboutGetOutMid(isRight)
            self.getTarget()
            self.solve()
            if midYaw * self.landmark["Yaw"] < 0:
                self.landmark["Yaw"] = midYaw

        isRight = False
        pred = 0
        self.pred = 0
        lastYaw  = -self.K
        # if testF2:
        #     pred,p_last,x_last = self.kalman(self.landmark["Yaw"],x_last,p_last,Q,R)
        #     self.pred = pred
        #     self.landmark["Yaw"] = pred
        if rFlag == 0:
            turnFlag = 0
        self.landmark["Yaw"] = self.landmark["Yaw"] - (M - 1 - self.right - self.left) * midK

        if lineF3 == 2:
            lineF3 = 0
        if lineF2 >= 1 and setlineF == 1:
            if self.checkStartLine(lineCheck3):
                lineF3 = 1
            if lineF3 == 1:
                lineC = lineC + 1
                self.landmark["Yaw"] = -2
            if lineC > lineCheck4:
                lineF2 = 0
                lineC = 0
                lineF3 = 2
        elif self.checkStartLine(lineCheck1) and forkFlag == 0 and wFlag == 0 and lineSet == 1:
            lineF1 = 1
        elif (lineF1 == 1 or lineF1 == 2):
            if abs(self.landmark["Yaw"]) > lineYaw:
                self.landmark["Yaw"] = 0
            if self.checkStartLine(lineCheck2):
                lineF1 = 2
            if (not self.checkStartLine(lineCheck2)) and lineF1 == 2:
                lineF1 = 0
                lineF2 = 1

        # 串口通信
        if ser.isOpen():
            str = struct.pack('f',self.landmark["Yaw"]) # 返回偏角
            ser.write('aa'.encode('utf-8'))
            ser.write(str)
            str2 = struct.pack('f',self.F / Fcheck1) # 返回弯道位置0-1，越靠近1表示离弯道越近
            ser.write(str2)
            if self.S <= Scheck and rFlag == 0:
                ser.write('i'.encode('utf-8')) # 长直道
            elif (self.F >= Fcheck1 and not Kflag) or not rFlag == 0 or forkFlag == forkWan or wC >= wCset3 or lineF3 == 1:
                ser.write('n'.encode('utf-8')) # 靠近弯道
            elif self.F >= Fcheck2:
                ser.write('a'.encode('utf-8')) # 较为靠近弯道
            else:
                ser.write('f'.encode('utf-8'))
            if lineF3 == 2:
                ser.write('e'.encode('utf-8'))
            elif rFlag == 0 or turnFlag == 1:
                ser.write('o'.encode('utf-8')) # 非环
            elif self.C >= CcheckD and self.C <= CcheckU:
                ser.write('t'.encode('utf-8')) # 开始转弯
                turnFlag = 1
            else:
                ser.write('u'.encode('utf-8')) # 环内
            ser.write('bb'.encode('utf-8'))

__all__ = ["ImgProcess"]
