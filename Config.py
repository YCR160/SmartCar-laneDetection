import cv2
import numpy as np
# 图像
N, M = 80, 176  # 图片的高和宽
CUT = 1# 裁剪最上面的多少行
PADDING = 1  # 舍弃左右边界的大小
CORNERCUT = 40  # 搜索前沿时舍弃最上面角的宽度

FORKUPCUT = 2  # 三岔路口前沿点最小有效行
FORKDOWNCUT = 13  # 三岔路口前沿点最大有效行
FORKLOW = 90  # 三岔路口最小角度
FORKHIGH = 160  # 三岔路口最大角度
FORKMAXDIST2 = 25

# sobel
LRSTEP = 2
UDSTEP = 1
THRESHLOD = 230  # sobel的阈值

# 用于排除点的斜率范围
K_LOW = -1.5 # 斜率下限
K_HIGH = 1.5  # 斜率上限

# 拟合曲线
X_POS = 102  # 平移点的x位置
WIDTH = 20  # 赛道宽度

# 获取目标点
PI = 144.0
DIST = 37  # 垂足向上扩展的长度

light = 190

blackN = 35

redF = 1
redCheck = 1000

wWan = 1
wFcheck = 60
wYawOut = 2
wYaw1 = -2
wYaw2 = 2
wYaw3 = 2.5
wYaw4 = -3
wF = 0
wCset1 = 17
wCset2 = 36
wCset3 = 36
wCset4 = 52
wCset5 = 64
wLeft = 11

forkWan = 1
fF = 0
forkCset1 = 14 #进入三岔右打帧数 inFork
forkCset2 = 25
forkCset3 = 0 #outFrok
forkCset4 = 0
forkYaw = 1.3
forkYawOut = -0.2
forkYawRange = 0.4
forkIRange = 15
forkK = 0.4
forkRed = 30

kn = 3
outF1 = 5 #出环标志
searchKn = 2
numberCheck = 1 #查找的斜率数量
numberF = 2 #找弯道系数
Fcheck1 = 110 #靠近弯道
Fcheck2 = 200 #较为靠近弯道
FitC = 0 #直道拟合距离
Scheck = 20 #直道计数
Scheck2 = 10 #置信数量
CcheckD = 5 #进环位置
CcheckU = 30
Kcheck = -1.5#出弯斜率
testF1 = 1
testF2 = 0
gauss = 0
cannydown = 150
cannyup = 250
Q = 0.1
R = 2
sleeptime = 0
midK = 0.005 #回中系数
length = 30 #前瞻
totC = 20 #最少许检测数量
getM = 5 #寻中线阈值

X0 = 27.0
S = 5
YAW = 1.6

# 起跑线检测
STARTLINE_I1 = 10
STARTLINE_I2 = 79
STARTLINE_PADDING = 30
STARTLINE_COUNT = 15
lineCheck1 = 20
lineCheck2 = 70
lineCheck3 = 20
lineCheck4 = 18
setlineF = 1
lineI = 10
lineYaw = 0.3
lineSet = 1

# 坡道
HILL_DIFF = 15
HILL_CUT = 30
HILL_COUNT = 10

# 环岛
ROUND_MAXWIDTH = 70  # 最大有效宽度，大于这个宽度视为丢线
ROUND_COUNT0 = 13  # 最开始至少丢的行数，设成0就可以不丢
ROUND_DIST2 = 8  # 圆形结束后最丢线的最小距离
ROUND_COUNT3 = 3  # 再次搜到线的最小行数

ROUND_UPCOUNT = 8  # 原图上圆环边先变小的最小个数
ROUND_DOWNCOUNT = 3  # 原图上圆环边变小后变大的最小个数

ROUNDENTER_GETCOUNT = 3  # 入环时从上往下搜的最少有效点
ROUNDENTER_LOSTCOUNT = 3  # 搜到有效点后再至少丢的行数

inRoundYaw = 1.3
roundCheck = 40
roundCheck0 = 10
# 逆透视变换
SRCARR = [  # 原图上的四个点
    (-5, 0),  # 左上角
    (-5, 174),  # 右上角
    (78, 0),  # 左下角
    (78, 174),  # 右下角
]
PERARR = [  # 新图上的四个点
    (4, 20),  # 左上角
    (4, 215),  # 右上角
    (125, 100),  # 左下角
    (125, 140),  # 右下角
]

# 可视化
SRCZOOM = 5  # 原图放大倍数
N_, M_ = 130, 235  # 新图的高和宽
I_SHIFT = -30  # 新图向下平移
J_SHIFT = 0  # 新图向右平移
PERZOOM = 4  # 新图放大倍数
COLORS = ((255, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 127, 127), (127, 127, 0))  # 画点的颜色
SRC = np.array(SRCARR,dtype=np.float32)
SRC = SRC[:, [1 , 0]]
PER = np.array(PERARR,dtype=np.float32)
PER = PER[:, [1 , 0]]
H = cv2.getPerspectiveTransform(SRC, PER)
HH = cv2.getPerspectiveTransform(PER, SRC)
