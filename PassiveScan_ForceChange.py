from ctypes import *
from io import DEFAULT_BUFFER_SIZE
from dwfconstants import *
import time
import sys
import serial
import asyncio
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
import csv
import pprint
import pandas as pd

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
time_aus = 5            # 計測時間[s]
# dBref = 2e-5
dBref = 1
target = [85,80] # 設定するプローブ目標位置[mm]

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

# 聴取音の録音
def record(index, samplerate, fs, time, delta_ez):
    pa = pyaudio.PyAudio()
    data = []
    dt = 1 / samplerate
    
    # ストリームの開始
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=samplerate,
                     input=True, input_device_index=index, frames_per_buffer=fs)
    
    # フレームサイズ毎に音声を録音していくループ
    for i in range(int(((time / dt) / fs))):
        frame = stream.read(fs)
        data.append(frame)

        ser.write(str.encode(delta_ez+"\0"))

    # ストリームの終了
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    # データをまとめる処理
    data = b"".join(data)
    
    # データをNumpy配列に変換
    data = np.frombuffer(data, dtype="int16") # / float((np.power(2, 16) / 2) - 1)

    return data, i

# 受動スキャン
def passive_ausc():
    global accum_delta
    global delta
    global pre_delta
    global currentPos
    Kp = 0.5
    TI = 10000
    TD = 1
    peak = np.zeros(1)

    for num in range(len(target)):
        # 目標位置にプローブ移動
        for _ in range(0,200):
            try:
                time.sleep(0.02)
                dwf.FDwfAnalogInStatus(hdwf, False, None) 
                dwf.FDwfAnalogInStatusSample(hdwf, c_int(0), byref(voltage1))
                delta = vol2mm(voltage1.value) - target[num]
                accum_delta += delta
                inc_val = Kp*(delta+(1/TI)*accum_delta+TD*(delta-pre_delta));
                pre_delta = delta
                currentPos += inc_val

                delta_ez = str(round(currentPos*4096/27))
                print(delta_ez)
                ser.write(str.encode(delta_ez+"\0"))
                r = ser.read()
            except KeyboardInterrupt:
                print("Ctrl+Cで停止しました")
                break
        
        # 聴取音の録音    
        wfm, i = record(dev_index, samp_rate, chunk, time_aus, delta_ez)
        t = np.arange(0, chunk * (i+1) * (1 / samp_rate), 1 / samp_rate)

        # グラフ表示
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Sound pressure [Pa]')
        ax1.plot(t, wfm, label='signal', lw=1)
        fig.tight_layout()
        plt.show()
        plt.close()

        # フーリエ変換
        fft_data = np.abs(np.fft.rfft(wfm))
        freqList = np.fft.rfftfreq(len(wfm), 1.0 / samp_rate)
        
        # グラフ表示
        plt.plot(freqList, 20 * np.log10(fft_data))
        plt.xscale('log')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.show()
        
        # 生データの保存
        tmp = np.vstack((np.array(wfm),np.array(t)))
        fileName = 'data_' + str(num) + '.csv'
        pd.DataFrame(tmp.T).to_csv(fileName)


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

    ser = serial.Serial('COM3',9600,timeout=0.1)

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

    callback = passive_ausc
    handler(callback)
