"""
最开始一个主函数，用来开两个进程，这两个进程分别用来进行主程序运行和显示调试图像
"""
import multiprocessing as mp
from os.path import dirname, realpath
from scripts.getConfig import getConfig
from scripts.Main import Main
import cv2
def work(DealFlag, ImgQueue): #主运行进程
    dir_ = dirname(realpath(__file__))
    Config = getConfig(dir_)
    main = Main(Config, DealFlag, ImgQueue)
    main.mainloop()
def show(DealFlag, ImgQueue): #调试输出进程，开两个进程的原因是输出图像这里需要一定延时，会导致性能下降，所以分开写
    while True:
        if DealFlag.value and not ImgQueue.empty():
            Img = ImgQueue.get()
            DealFlag.value = 0
            cv2.imshow('1', Img)
            cv2.waitKey(5)
if __name__ == "__main__":
    ImgQueue = mp.Queue()  # 先进先出队列，实现不同进程数据交互
    DealFlag = mp.Value('B', 0)  # DealFlag为1，则表示有图片，可处理
    Mps = []
    Mps.append(mp.Process(target=work, args=(DealFlag, ImgQueue)))
    Mps.append(mp.Process(target=show, args=(DealFlag, ImgQueue)))
    [Mp.start() for Mp in Mps]
    Mps[0].join()
    [Mp.terminate() for Mp in Mps]
