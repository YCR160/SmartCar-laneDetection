from .Main import Main
from Config import *


class ImgWindow:
    """
    用于对视频帧进行处理，并在处理后的视频帧上添加文本标签
    """

    def __init__(self, main: Main) -> None:
        from .ImgProcess import ImgProcess
        self.main = main
        self.imgProcess = ImgProcess()
        self.setImg = self.imgProcess.setImg