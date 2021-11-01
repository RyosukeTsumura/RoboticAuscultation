from ctypes import *
from dwfconstants import *
import time
import sys
import serial
import asyncio
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

global delta
global target
global currentPos
global accum_delta
global pre_delta

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz　サンプリング周波数
N = 1
chunk = 1024*N # 一度に取得するデータ数
dev_index = 1 # デバイス番号
sleepTime = 0.0001

dBref = 2e-5

# dBへの変換
def db(x, dBref):
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 20 * np.log10(x / dBref)     #変換式
    return y                         #dB値を返

def vol2mm(vol):
    dist = 10*vol+100
    return dist

def handler(func, *args):
    return func(*args)

def move():
    global accum_delta
    global delta
    global pre_delta
    global currentPos
    # Kp = 0.1
    # TI = 2000
    # TD = 1
    Kp = 0.5
    TI = 10000
    TD = 1

    for _ in range(1200):
        time.sleep(0.02)
        dwf.FDwfAnalogInStatus(hdwf, False, None) 
        dwf.FDwfAnalogInStatusSample(hdwf, c_int(0), byref(voltage1))

        delta = vol2mm(voltage1.value) - target
        accum_delta += delta
        inc_val = Kp*(delta+(1/TI)*accum_delta+TD*(delta-pre_delta));
        pre_delta = delta
        currentPos += inc_val

        delta_ez = str(round(currentPos*4096/27))
        print(delta_ez)
        ser.write(str.encode(delta_ez+"\0"))
        r = ser.read()
        print(r)

    time.sleep(1)
    dwf.FDwfDeviceCloseAll()
    ser.write(str.encode('0'))
    ser.close()

def passive_ausc():
    global accum_delta
    global delta
    global pre_delta
    global currentPos
    # Kp = 0.1
    # TI = 2000
    # TD = 1
    Kp = 0.5
    TI = 10000
    TD = 1

    # for _ in range(1200):
    #     time.sleep(0.02)
    #     dwf.FDwfAnalogInStatus(hdwf, False, None) 
    #     dwf.FDwfAnalogInStatusSample(hdwf, c_int(0), byref(voltage1))

    #     delta = vol2mm(voltage1.value) - target
    #     accum_delta += delta
    #     inc_val = Kp*(delta+(1/TI)*accum_delta+TD*(delta-pre_delta));
    #     pre_delta = delta
    #     currentPos += inc_val

    #     delta_ez = str(round(currentPos*4096/27))
    #     print(delta_ez)
    #     ser.write(str.encode(delta_ez+"\0"))
    #     r = ser.read()
    #     print(r)

    while True:
        try:
            # 音声データの取得
            data = stream.read(chunk)
            ndarray = np.frombuffer(data, dtype='int16')
            
            # FFT
            wave_y = np.fft.fft(ndarray)
            wave_y = np.abs(wave_y)

            # dB変換
            # wave_y = db(np.sqrt(wave_y),dBref)

            # データ整理
            wave_y2 = wave_y[0:chunk2]
            
            # ピーク検出処理
            # peak = max(wave_y2)
            # print('ピークdB:',peak)
            # print('ピーク周波数：', wave_x2[np.argmax(wave_y2)])

            # グラフ表示
            plt.plot(wave_x2,wave_y2)
            plt.draw()
            plt.pause(sleepTime)
            plt.cla()

            dwf.FDwfAnalogInStatus(hdwf, False, None) 
            dwf.FDwfAnalogInStatusSample(hdwf, c_int(0), byref(voltage1))

            delta = vol2mm(voltage1.value) - target
            accum_delta += delta
            inc_val = Kp*(delta+(1/TI)*accum_delta+TD*(delta-pre_delta));
            pre_delta = delta
            currentPos += inc_val

            delta_ez = str(round(currentPos*4096/27))
            print(delta_ez)
            ser.write(str.encode(delta_ez+"\0"))
            r = ser.read()
            print(r)

        except KeyboardInterrupt:
            print("Ctrl+Cで停止しました")
            break


    time.sleep(1)
    dwf.FDwfDeviceCloseAll()
    ser.write(str.encode('0'))
    ser.close()

if __name__ == '__main__':

    """
    Config_MigthyZap
    """
    currentPos = 0
    delta = 0
    accum_delta = 0
    pre_delta = 0
    # target = 112.5
    target = 85
    ser = serial.Serial('COM12',9600,timeout=0.1)

    """
    Config_Analog Discovery 2
    """

    if sys.platform.startswith("win"):
        dwf = cdll.dwf
    elif sys.platform.startswith("darwin"):
        dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
    else:
        dwf = cdll.LoadLibrary("libdwf.so")

    #declare ctype variables
    hdwf = c_int()
    voltage1 = c_double()
    voltage2 = c_double()

    #print(DWF version
    version = create_string_buffer(16)
    dwf.FDwfGetVersion(version)
    print("DWF Version: "+str(version.value))

    #open device
    "Opening first device..."
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

    if hdwf.value == hdwfNone.value:
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        print(szerr.value)
        print("failed to open device")
        quit()

    print("Preparing to read sample...")
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True)) 
    dwf.FDwfAnalogInChannelOffsetSet(hdwf, c_int(0), c_double(0)) 
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5)) 
    dwf.FDwfAnalogInConfigure(hdwf, c_bool(False), c_bool(False)) 

    time.sleep(2)

    audio = pyaudio.PyAudio() # create pyaudio instantiation

    # ストリームの作成
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)

    # x軸
    wave_x = np.linspace(0, samp_rate, chunk)
    chunk2 = int(chunk/2)
    wave_x2 = wave_x[0:chunk2]

    callback = passive_ausc
    handler(callback)

    # i = 0
    # while i < 10:
    #     time.sleep(0.5)
    #     dwf.FDwfAnalogInStatus(hdwf, False, None) 
    #     dwf.FDwfAnalogInStatusSample(hdwf, c_int(0), byref(voltage1))
    #     # dwf.FDwfAnalogInStatusSample(hdwf, c_int(1), byref(voltage2))
    #     # print("Channel 1:  " + str(voltage1.value)+" V")
    #     # print("Channel 2:  " + str(voltage2.value)+" V")
    #     dist = vol2mm(voltage1.value) -110
    #     # print("Channel 1:  " + str(dist)+" mm")
    #     dist_ez = str(round(dist*4096/27))
    #     print(dist_ez)
    #     # ser.write(str.encode(str(0)))
    #     ser.write(str.encode(dist_ez))
    #     r = ser.read()
    #     print(r)
    #     i += 1

        
    # time.sleep(1)
    # dwf.FDwfDeviceCloseAll()
    # ser.write(str.encode('0'))
    # ser.close()
