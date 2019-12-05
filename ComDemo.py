# -*- coding: utf-8 -*
import serial
import binascii
import time
# 打开串口
ser = serial.Serial("/dev/ttyAMA0", 57600)
def recvser():
    count = ser.inWaiting()# 获得接收缓冲区字符
    if count != 0:
        recv = ser.read(count)
        hlen = len(recv)
        recvv = ""
        for i in xrange(hlen):
            hvol = ord(recv[i])
            hhex = '%02x' % hvol

            recvv += hhex + ''
        print recvv
        ser.flushInput()#
        time.sleep(0.1)# 必要的软件延时
        return recvv
    else:
        time.sleep(0.1)
        return 0
def writeser(datt):
    hexstr = binascii.a2b_hex(datt)

    ser.write(hexstr)

def main():
    while True:
        # print '
        # writeser("AA2BBB")
        # print 'stop'
        # time.sleep(3)
        # writeser("AA1100")# 写数据
        # print 'auto'
        # time.sleep(3)
        # time.sleep(0.1)
        recvser()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if ser != None:
            ser.close()