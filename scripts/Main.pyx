"视频处理"
import cv2
import time

capture = cv2.VideoCapture(0)
# print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
capture.set(cv2.CAP_PROP_FRAME_WIDTH,176)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,144)
# print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,30.0,(176,80))

class Main:
    def __init__(self, Config) -> None:
        from .ImgWindow import ImgWindow
        co = 3000
        # 帧大小缩小到宽度为 176 像素，高度为 80 像素
        down_width, down_height = 176, 80
        down_points = (down_width, down_height)

        while True:
            print(co)
            co = co - 1
            start = time.time()
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
            output = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.Config = Config
            self.imgWindow = ImgWindow(self)
            self.imgWindow.setImg(frame)
            self.imgWindow.imgProcess.work()

            output = cv2.putText(output,str(self.imgWindow.imgProcess.landmark["Yaw"]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            end = time.time()
            print(1/(end-start))

            # if cv2.waitKey(2) == ord('q'):
            #   break

        capture.release()
        out.release()
