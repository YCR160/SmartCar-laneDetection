from .Main import Main
from Config import *


class ImgWindow:
    def __init__(self, main: Main) -> None:
        from .ImgProcess import ImgProcess
        self.main = main
        self.imgProcess = ImgProcess()
        self.setImg = self.imgProcess.setImg