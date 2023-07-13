"""
最开始一个主函数，用来开两个进程，这两个进程分别用来进行主程序运行和显示调试图像
可以再开两个进程，分别用于模型识别和模型图像传输
"""
import multiprocessing as mp
import os
from scripts.getConfig import getConfig
from scripts.Main import Main
import cv2

# 以下代码引入PaddleDetection模块
import sys
sys.path.append('../PaddleLiteDemo/Python/')
import detection as Det
import time

# shared_capture = cv2.VideoCapture(0)
# from scripts.Main import Main

def work(DealFlag, ImgQueue, Frame, DetectionResult):
    """
    主运行进程
    :param DealFlag: 共享内存变量，用于标记是否有图像可处理
    :param ImgQueue: 用于存储待处理图像队列
    :return: None
    """
    # 获取当前脚本文件的绝对路径
    full_path = os.path.dirname(os.path.realpath(__file__))
    Config = getConfig(full_path)
    img_processor = Main(Config, DealFlag, ImgQueue, Frame, DetectionResult)
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

def detection(Frame, DetectionResult):
    system_config_path = "../PaddleLiteDemo/configs/detection/ssd_mobilenet_v1/usbcamera.json"
    print("SystemConfig Path: {}".format(system_config_path))
    Det.g_system_config = Det.SystemConfig(system_config_path)
    model_config_path = Det.g_system_config.model_config_path
    system_config_root = system_config_path[:system_config_path.rfind("/")]
    Det.g_model_config = Det.ModelConfig(os.path.join(system_config_root, model_config_path))

    display = Det.Display("PaddleLiteDetectionDemo")
    timer = Det.Timer("Predict", 100)
    # capture = Det.createCapture(Det.g_system_config.input_type, Det.g_system_config.input_path)
    # capture.cap = shared_capture
    # if capture is None:
    #     exit(-1)

    ret = Det.predictorInit()
    if ret != 0:
        print("Error!!! predictor init failed .")
        sys.exit(-1)

    while True:
        # frame = capture.getFrame()
        # Frame.put(frame)
        # print(len(frame[0]))

        frame_flag = 0
        while frame_flag == 0:
            while not Frame.empty():
                frame = Frame.get()
                frame_flag = 1

        origin_frame = frame.copy()
        start = time.time()
        predict_result = Det.predict(frame, timer)
        end = time.time()
        DetectionResult.put(predict_result)
        # print("result_len", len(predict_result))
        # for result in predict_result:
        #     print("result_index: ", result.type)
        #     print("result_x: ", result.x)
        #     print("result_y: ", result.y)
        #     print("result_h: ", result.height)
        #     print("result_w: ", result.width)
        # print("result",predict_result, end-start)
        # print("frame_size: ", origin_frame.size)
        # Det.drawResults(origin_frame, predict_result)
        if Det.g_system_config.predict_log_enable:
            Det.printResults(origin_frame, predict_result)

        # if Det.g_system_config.predict_log_enable and capture.getType() != "image":
        #     display.putFrame(origin_frame)
        # if capture.getType() == "image":
        #     cv2.imwrite("DetectionResult.jpg", origin_frame)
        #     break

        # if Det.g_system_config.predict_time_log_enable:
        #     timer.printAverageRunTime()

    if Det.g_system_config.display_enable and Det.g_system_config.input_type != "image":
        display.stop()

    # capture.stop()
    return


if __name__ == "__main__":
    ImgQueue = mp.Queue()           # 用于存储待处理的图像，先进先出队列，实现不同进程数据交互
    DealFlag = mp.Value('B', 0)     # DealFlag为 1 则表示有待处理图片，为 0 则表示没有待处理图片
    Frame = mp.Queue()              # 用于存储原始图像，实现不同进程数据交互
    DetectionResult = mp.Queue()    # 用于存储检测结果，实现不同进程数据交互
    Mps = []                        # 用于存储多个进程对象

    Mps.append(mp.Process(target=work, args=(DealFlag, ImgQueue, Frame, DetectionResult)))
    Mps.append(mp.Process(target=show, args=(DealFlag, ImgQueue)))
    Mps.append(mp.Process(target=detection, args=(Frame, DetectionResult)))

    # 列表推导式启动 Mps 列表中的所有进程
    [Mp.start() for Mp in Mps]
    # 等待第一个进程结束
    Mps[0].join()
    # 列表推导式终止 Mps 列表中的所有进程
    [Mp.terminate() for Mp in Mps]

    # shared_capture.release()
