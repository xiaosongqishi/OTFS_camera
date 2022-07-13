import socket
import cv2
import threading
import struct
import numpy
from PIL import Image


class Camera_Connect_Object:
    def __init__(self, D_addr_port=["", 8880]):
        self.resolution = [320, 240]
        self.addr_port = D_addr_port
        self.src = 888 + 15  # 双方确定传输帧数，（888）为校验值
        self.interval = 0  # 图片播放时间间隔
        self.img_fps = 15  # 每秒传输多少帧数

    def Set_socket(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def Socket_Connect(self):
        self.Set_socket()
        self.client.connect(self.addr_port)
        print("IP is %s:%d" % (self.addr_port[0], self.addr_port[1]))

    def RT_Image(self):
        # 按照格式打包发送帧数和分辨率
        self.name = self.addr_port[0] + " Camera"
        data = struct.pack("ihh", self.src, self.resolution[0], self.resolution[1])
        print("datalen:", len(data), data)
        info = struct.unpack("ihh", data)
        print(info[0], info[1], info[2])
        self.client.send(data)
        print("after send")
        # file=open('data.txt','w')
        # print("file open")
        # file.write('get')
        # file.close()
        # print("file close")
        while (1):
            info = struct.unpack("ihh", self.client.recv(8))
            print("datalen:", info[0])
            buf_size = info[0]  # 获取读的图片总长度
            if buf_size:
                try:
                    self.buf = b""  # 代表bytes类型
                    temp_buf = self.buf
                    while (buf_size):  # 读取每一张图片的长度
                        temp_buf = self.client.recv(buf_size)
                        buf_size -= len(temp_buf)
                        self.buf += temp_buf  # 获取图片
                    data = numpy.frombuffer(self.buf, dtype='uint8')  # 按uint8转换为图像矩阵
                    # file = open('data.txt', 'a')
                    # print("file open")
                    # file.write(data +'\n')
                    # print("file write")
                    # file.close()
                    # print("file close")
                    self.image = cv2.imdecode(data, 1)  # 图像解码
                    cv2.imwrite('img.jpg', self.image)
                    # print(type(self.image))  # 返回Image对象
                    # print(self.image.size)  # 返回图像的尺寸
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    cv2.imshow(self.name, self.image)  # 展示图片


                except:
                    pass;
                finally:
                    if (cv2.waitKey(20) == 81):  # 每10ms刷新一次图片，按‘ESC’（27）退出
                        self.client.close()
                        cv2.destroyAllWindows()
                        break

    def Save_Image(self):
        file = open('data.txt', 'w')
        print("file open")
        file.write('get\n')
        file.close()
        print("file close")

    def Get_Data(self, interval):
        showThread = threading.Thread(target=self.RT_Image)
        showThread.start()

    def Save_Data(self, interval):
        showThread = threading.Thread(target=self.Save_Image)
        showThread.start()


if __name__ == '__main__':
    camera = Camera_Connect_Object()
    camera.addr_port[0] = "222.20.75.71"  # 服务端的IP地址
    #camera.addr_port[0] = "192.168.56.1"
    # input("Please input IP:")
    camera.addr_port = tuple(camera.addr_port)
    camera.Socket_Connect()
    camera.Get_Data(camera.interval)
    # camera.Save_Data(camera.interval)
