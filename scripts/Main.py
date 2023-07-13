"""
主函数后进来的第一个函数，在这个函数里面定义了摄像头相关内容，获取摄像头图像交给函数处理，同时得到处理函数的返回值用于调试输出
"""
import cv2
import time
import numpy as np
import multiprocessing as mp
from Config import *

capture = cv2.VideoCapture(0)
# # print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# capture.set(cv2.CAP_PROP_CONTRAST, 30)  # 30 对比度
# capture.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 20 亮度
# # 通过调整对比度和亮度，让图像能够稳定下来，不会在运行过程中因为亮度变化而导致鲁棒性变差
# # print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# # print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (176, 80))


class Main:
    # def show(self, DealFlag, ImgQueue):

    def __init__(self, Config, DealFlag, ImgQueue, Frame, DetectionResult) -> None:
        from .ImgWindow import ImgWindow
        from .ResultProcess import ResultProcess

        self.Config = Config
        self.resultProcess = ResultProcess()

        # 帧大小缩小到宽度为 176 像素，高度为 80 像素
        down_width, down_height = 176, 80
        down_points = (down_width, down_height)

        alpha = 1.5  # 对比度调整系数
        beta = 30  # 亮度调整系数

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            Frame.put(frame)

            # frame = 0
            # frame_flag = 0
            # while frame_flag == 0:
            #     while not Frame.empty():
            #         frame = Frame.get()
            #         frame_flag = 1

            frame = cv2.convertScaleAbs(
                frame, alpha=alpha, beta=beta)  # 调整亮度和对比度
            frame = np.array(frame)
            frame = cv2.resize(frame, down_points,
                               interpolation=cv2.INTER_LINEAR)

            detectionresult = 0
            detectionresult_flag = 0
            # 取出最后一个识别结果
            while not DetectionResult.empty():
                detectionresult = DetectionResult.get()
                detectionresult_flag = 1

            self.imgWindow = ImgWindow(self)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
            self.imgWindow.setImg(frame)  # 将图像传入 ImgWindow

            if detectionresult_flag == 1:
                self.resultProcess.update(detectionresult)

            self.imgWindow.imgProcess.work(
                self.resultProcess.getResult())  # 将识别结果传入 ImgProcess

            # 后面是一些调试输出
            output = self.imgWindow.imgProcess.canny2
            color = (0, 0, 255)  # 蓝色
            # print("F========================", self.imgWindow.imgProcess.F)

            # S 值小于等于 Scheck，则将标记颜色设置为黄色
            if self.imgWindow.imgProcess.S <= Scheck:
                color = (255, 255, 0)
            # F 值大于等于 Fcheck1，且 Kflag 标志位为 False，则将标记颜色设置为红色
            elif self.imgWindow.imgProcess.F >= Fcheck1 and not self.imgWindow.imgProcess.Kflag:
                color = (255, 0, 0)
            # F 大于等于 Fcheck2，则将标记颜色设置为绿色
            elif self.imgWindow.imgProcess.F >= Fcheck2:
                color = (0, 255, 0)
            # 在图像上绘制线段
            cv2.line(output, (176 // 2 - 1, 80 - 1), (round(176 // 2 - 1 +
                     40 * self.imgWindow.imgProcess.landmark["Yaw"]), 80-40), color, 2)
            # 用于将处理后的视频帧存储在一个队列中，并将队列交给另一个进程进行输出
            # 如果共享变量 DealFlag 为 0
            if not DealFlag.value:
                # 将队列中的所有元素取出，并将处理后的视频帧存储在队列中
                while not ImgQueue.empty():
                    ImgQueue.get()
                ImgQueue.put(output)
                DealFlag.value = 1

            # cv2.imshow('2',self.imgWindow.imgProcess.PERMAT)
            # cv2.waitKey(2)
            cv2.waitKey(1)
            # output = cv2.putText(output,str(self.imgWindow.imgProcess.landmark["Yaw"]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # out.write(self.imgWindow.imgProcess.canny2)
            # process = mp.Process(target=show, args=(DealFlag, ImgQueue))
            # if cv2.waitKey(2) == ord('q'):
            #    break

        # capture.release()
        # out.release()
