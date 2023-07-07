import cv2
import time
capture = cv2.VideoCapture(0)
#print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
capture.set(cv2.CAP_PROP_FRAME_WIDTH,176)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,144)
#print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,30.0,(176,80))
class Main:
    def __init__(self, Config) -> None:
        from .ImgWindow import ImgWindow
        co = 3000
        while True:
            print(co)
            co = co - 1
            start = time.time()
            ret, frame = capture.read()
            #print(frame.shape)
            if not ret:
                break
            #frame = frame[0:80, 0:176]
            # Set rows and columns 
            # lets downsize the image using new  width and height
            down_width = 176
            down_height = 80
            down_points = (down_width, down_height)
            frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
            output = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("1", frame)
            #cv2.waitKey(1)
            #out.write(frame)
            self.Config = Config
            self.imgWindow = ImgWindow(self)
            self.imgWindow.setImg(frame)
            self.imgWindow.imgProcess.work()
            output = cv2.putText(output,str(self.imgWindow.imgProcess.landmark["Yaw"]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #out.write(output)
            end = time.time()
            print(1/(end-start))
            #print(frame)
            #if cv2.waitKey(2) == ord('q'):
            #    break
        capture.release()
        out.release()
