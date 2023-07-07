"""
主函数后进来的第一个函数，在这个函数里面定义了摄像头相关内容，获取摄像头图像交给函数处理，同时得到处理函数的返回值用于调试输出
"""
import cv2
import time
import numpy as np
import multiprocessing as mp
#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from Config import *
capture = cv2.VideoCapture(0)
#print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
capture.set(cv2.CAP_PROP_FRAME_WIDTH,320) 
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
capture.set(cv2.CAP_PROP_CONTRAST, 30) #30 对比度
capture.set(cv2.CAP_PROP_BRIGHTNESS, 20) #20 亮度
#通过调整对比度和亮度，让图像能够稳定下来，不会在运行过程中因为亮度变化而导致鲁棒性变差
#print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,30.0,(176,80))
class Main:
    #def show(self, DealFlag, ImgQueue):

    def __init__(self, Config, DealFlag, ImgQueue) -> None:
        from .ImgWindow import ImgWindow
        co = 1000 #这里是录制的一个帧数计数
        a = []
        b = []
        lastwS = 0
        while True:
            #print(co)
            #co = co - 1
            start = time.time()
            ret, frame = capture.read()
            #print(frame.shape)
            if not ret or not co:
                break
            # Set rows and columns 
            # lets downsize the image using new  width and height
            down_width = 176
            down_height = 80
            down_points = (down_width, down_height)
            frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #使用HSV颜色空间
            lower_red = np.array([160,70,81])#设置需要识别的车道线颜色的阈值
            upper_red = np.array([179,255,255])
            lower_red2 = np.array([0,70,81])#设置需要识别的车道线颜色的阈值
            upper_red2 = np.array([20,255,255])
            #dark_blue = np.uint8([[[12,22,121]]])
            #dark_blue = cv2.cvtColor(dark_blue,cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv,lower_red,upper_red)+cv2.inRange(hsv,lower_red2,upper_red2) #提取所需颜色
            ker = np.ones((1,3), np.uint8)
            mask = cv2.erode(mask,ker)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(37, 10))
            mask = cv2.dilate(mask,kernel)
            countRed = np.count_nonzero(mask[50:79,:])
            countRed2 = np.count_nonzero(mask)
            print("reddddddddddddddddddddddddddd", countRed) #此处是为了识别赛道中的红色元素写的红色判断
            nonzero = np.nonzero(mask)
            #x = nonzero[0]
            #y = nonzero[1]
            if countRed2 > 50:
                frame[nonzero] = [220,220,220] #这里将赛道中心的红色标志给去掉，防止干扰后面的赛道识别
            fcolor1 = frame[79,87]
            fcolor2 = frame[60,87]
            fcolor1 = np.array(fcolor1)
            fcolor2 = np.array(fcolor2)
            #warped = cv2.warpPerspective(frame, H, (200, 200))
            #cv2.imshow('g',frame)
            #cv2.waitKey(1)
            self.Config = Config
            self.imgWindow = ImgWindow(self)
            #if lastwS >= 2:
            #    frame[:,:,0] = frame[:,:,0] & mask
            #    frame[:,:,1] = frame[:,:,1] & mask
            #    frame[:,:,2] = frame[:,:,2] & mask
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.imgWindow.setImg(frame)
            self.imgWindow.imgProcess.work(fcolor1,fcolor2,countRed, countRed2)
            #后面是一些调试输出
            lastwS = self.imgWindow.imgProcess.wS
            #a.append(self.imgWindow.imgProcess.landmark["Yaw"])
            #b.append(self.imgWindow.imgProcess.pred)
            output = self.imgWindow.imgProcess.canny2
            color = (0, 0, 255)
            #print("down",count)
            print("F========================",self.imgWindow.imgProcess.F)
            if self.imgWindow.imgProcess.S <= Scheck:
                color = (255, 255, 0)
            elif self.imgWindow.imgProcess.F >= Fcheck1 and not self.imgWindow.imgProcess.Kflag:
                color = (255,0,0)
            elif self.imgWindow.imgProcess.F >= Fcheck2:
                color = (0, 255, 0)
            cv2.line(output, (176 // 2 - 1,80 - 1), (round(176 // 2 - 1 + 40 * self.imgWindow.imgProcess.landmark["Yaw"]),80-40), color, 2)
            if not DealFlag.value: #将要输出的图片送入队列，交给另一个进程做输出
                while not ImgQueue.empty():
                    ImgQueue.get()
                ImgQueue.put(output)
                #ImgQueue.put(warped)
                DealFlag.value = 1
            #cv2.imshow('2',self.imgWindow.imgProcess.PERMAT)
            #cv2.waitKey(2)
            cv2.waitKey(1)
            #output = cv2.putText(output,str(self.imgWindow.imgProcess.landmark["Yaw"]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #out.write(self.imgWindow.imgProcess.canny2)
            #process = mp.Process(target=show, args=(DealFlag, ImgQueue))
            end = time.time()
            print(1/(end-start)) #计算帧率
            #print(frame)
            #if cv2.waitKey(2) == ord('q'):
            #    break
        #plt.plot(a,color="b")  #真实值
        #plt.plot(b,color="g")     #测量值
        #plt.show()
        capture.release()
        #out.release()
