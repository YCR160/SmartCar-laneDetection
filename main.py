"""
最开始一个主函数，用来开两个进程，这两个进程分别用来进行主程序运行和显示调试图像
可以再开两个进程，分别用于模型识别和模型图像传输
"""
import multiprocessing as mp
import os
from scripts.getConfig import getConfig
from scripts.Main import Main
import cv2


def work(DealFlag, ImgQueue):
    """
    主运行进程
    :param DealFlag: 共享内存变量，用于标记是否有图像可处理
    :param ImgQueue: 用于存储待处理图像队列
    :return: None
    """
    # 获取当前脚本文件的绝对路径
    full_path = os.path.dirname(os.path.realpath(__file__))
    Config = getConfig(full_path)
    img_processor = Main(Config, DealFlag, ImgQueue)
    # 不断从 ImgQueue 中获取待处理的图像，并调用 Main 对象的方法进行处理，直到 DealFlag 为 0
    img_processor.mainloop()


def show(DealFlag, ImgQueue):
    """
    它用于在窗口中显示待处理的图像，调试输出进程，开两个进程的原因是输出图像这里需要一定延时，会导致性能下降，所以分开写
    :param DealFlag: 共享内存变量，用于标记是否有图像可处理
    :param ImgQueue: 用于存储待处理图像队列
    :return: None
    """
    while True:
        if DealFlag.value and not ImgQueue.empty():
            Img = ImgQueue.get()
            # 重置 DealFlag 为 0，表示图像已经被处理
            DealFlag.value = 0
            cv2.imshow('1', Img)
            # 等待 5 毫秒，以允许窗口显示图像
            cv2.waitKey(5)


if __name__ == "__main__":
    ImgQueue = mp.Queue()           # 用于存储待处理的图像，先进先出队列，实现不同进程数据交互
    DealFlag = mp.Value('B', 0)     # DealFlag为 1 则表示有待处理图片，为 0 则表示没有待处理图片
    Mps = []                        # 用于存储多个进程对象

    Mps.append(mp.Process(target=work, args=(DealFlag, ImgQueue)))
    Mps.append(mp.Process(target=show, args=(DealFlag, ImgQueue)))

    # 列表推导式启动 Mps 列表中的所有进程
    [Mp.start() for Mp in Mps]
    # 等待第一个进程结束
    Mps[0].join()
    # 列表推导式终止 Mps 列表中的所有进程
    [Mp.terminate() for Mp in Mps]
