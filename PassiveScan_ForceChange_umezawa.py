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
import os

import glob
from natsort import natsorted

global delta
global target
global currentPos
global accum_delta
global pre_delta


#条件変更時は下のfilepathを要修正
filepath = "./data/test/"
FFTpath = filepath + "FFT/"
RAWpath = filepath + "RAW/"
os.makedirs(FFTpath, exist_ok=True)
os.makedirs(RAWpath, exist_ok=True)


form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz　サンプリング周波数
# N = 2
# chunk = 1024*N # 一度に取得するデータ数
chunk = 2**11
dev_index = 1 # デバイス番号
sleepTime = 0.0001
time_aus = 5            # 計測時間[s]
# dBref = 2e-5
dBref = 1
# target = [85,80] # 設定するプローブ目標位置[mm]
# targetN = [0.5,1.0,1.5,2.0,2.5,3.0]  #設定するプローブ目標接触力a.a[N]
targetN = [3.5]  #設定するプローブ目標接触力a.a[N]


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

def springconst(x):
    F = 0.834*abs(85-x)    
    return F

def Ntomm(F):
    x = 85 - F / 0.834
    return x

# 聴取音の録音
def record(index, samplerate, fs, time, delta_ez):
    pa = pyaudio.PyAudio()
    data = []
    dt = 1 / samplerate
    
    # ストリームの開始
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=samplerate,
                     input=True, input_device_index=index, frames_per_buffer=fs)
    
    # フレームサイズ毎に音声を録音していくループ
    # for i in range(int(((time / dt) / fs))):
    for i in range(0,int(samplerate / chunk * time)):
        frame = stream.read(fs)
        data.append(frame)

        ser.write(str.encode(delta_ez+"\0"))

    # ストリームの終了
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    # データをまとめる処理
    data = b''.join(data)
    
    # データをNumpy配列に変換
    data = np.frombuffer(data, dtype="int16") # / float((np.power(2, 16) / 2) - 1)
    data = data / 32768

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
    # target = list(map(int,list(map(Ntomm , targetN))))
    target = list(map(Ntomm , targetN))
    print(target)

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
        # t = np.arange(0, chunk * (i+1) * (1 / samp_rate), 1 / samp_rate)
        t = range(len(wfm))
        args = sys.argv



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
        ax1.plot(t, wfm/max(wfm), label='signal', lw=1)
        fig.tight_layout()
        plt.show()
        plt.close()

        # フーリエ変換
        fft_data = np.abs(np.fft.fft(wfm))
        # freqList = np.fft.rfftfreq(len(wfm), 1.0 / samp_rate)
        freqList = np.fft.fftfreq(wfm.shape[0], d=1.0/samp_rate)  

        
        # グラフ表示
        # plt.plot(freqList, 20 * np.log10(fft_data))
        plt.plot(freqList[:int(len(freqList)/2)], fft_data[:int(len(freqList)/2)], label='sample', lw=1)
        
        # plt.xscale('log')
        
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.xlim(0,1000)
        plt.show()
       
              
        # 生データの保存
        tmp = np.vstack((np.array(wfm),np.array(t)))
        # fileName = filepath + 'data_' + str(num) + '.csv'
        fileName = RAWpath + str(targetN[num]) + 'N_RAW.csv'
        pd.DataFrame(tmp.T).to_csv(fileName)
        
        #フーリエ変換結果の保存
        outfile = open(FFTpath + str(targetN[num]) + 'N_FFT.csv','w', newline='')
        writer = csv.writer(outfile)
        writer.writerow(['freq', 'amplitude'])
        for i in range(int(len(freqList)/2)):
            writer.writerow([freqList[i],fft_data[i]])
            if freqList[i] >= 1000:
                break
        outfile.close()
        


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

    ser = serial.Serial('COM10',9600,timeout=0.1)

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
    
    
    #最適荷重導出
    basefile = './data/test/FFT/0.5N_FFT.csv'
    file = FFTpath
    allFiles = glob.glob(file + "/*.csv")
    list_ = []
    graphname_ = []
    allFiles = natsorted(allFiles)
    print(allFiles)

    baseframe = pd.read_csv(basefile,encoding="SHIFT_JIS")
    # baseframe['aveamplitude'] = baseframe['amplitude'].rolling(40, center=True).mean()
    baseframe['aveamplitude'] = baseframe['amplitude']

    # baseframe['amplitude'] = 10 * numpy.log10(baseframe['amplitude']^2)


    for file_ in allFiles:
        df = pd.read_csv(file_,encoding="SHIFT_JIS")#,index_col=None, header=0) # エクセルをデータフレームとして読み込む
        # df['amplitude'] = 10 * numpy.log10(df['amplitude']^2)
        # df['aveamplitude'] = df['amplitude'].rolling(40, center=True).mean()
        df['aveamplitude'] = df['amplitude']
        df['atten'] = df['aveamplitude'] / baseframe['aveamplitude'] * 100   #透過率
        # df['atten'] = df['aveamplitude'] - baseframe['aveamplitude']           #減衰    
        # df = df[::20]
        list_.append(df)
        # graphname_.append(re.findall(r'\d+',file_)[-1])
        graphname_.append(file_[-12:-9])
        # df['atten'] = df['amplitude'] - baseframe['amplitude']
        # df.to_csv("./filterout/" + re.findall(r'\d+',file_)[-1] + "test.csv")
    # frame = pd.concat(list_) # joinをinnerに指定

    # print(allFiles)
    print(graphname_)

    # print(list_)

    # print(baseframe)
    
    
    
    # FFTsums = []

    # # 羅列
    # fig, ax = plt.subplots(nrows=2,ncols=4) 
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # for i,x in enumerate(list_):
    #     if i==7: break
    #     j=i//4
    #     k=i%4

    #     ax[j,k].plot(x['freq'],x['aveamplitude'])
    #     # ax[j,k].plot(baseframe['freq'],baseframe['aveamplitude'])
    #     goukei = x['aveamplitude'].sum()
    #     goukei = round(goukei)
    #     FFTsums.append(goukei)
    #     # ax[j,k].text(50,1934,"振幅スペクトルの合計" + str(goukei) ,backgroundcolor="#ffffff",fontname="MS Gothic")
    #     print(goukei)
    #     gensui = (1-x['aveamplitude'].sum() / baseframe['aveamplitude'].sum())*100
    #     gensui = round(gensui)
    #     # ax[j,k].text(130,1734,"減衰率:" + str(gensui) + "%",backgroundcolor="#ffffff",fontname="MS Gothic")
    #     ax[j,k].text(190,1934,"振幅スペクトルの合計:" + str(goukei) + "\n減衰率:" + str(gensui) + "%",backgroundcolor="#ffffff",fontname="MS Gothic", horizontalalignment="right")


    #     ax[j,k].set_xlim([0, 200])
    #     ax[j,k].set_ylim([0, 2500])
    #     # ax[j,k].set_ylim([1, 40])
    #     ax[j,k].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
    #     ax[j,k].set_ylabel("振幅スペクトル", size = 14, weight = "light",fontname="MS Gothic")
    #     # ax[j,k].set_yscale("log")
    #     ax[j,k].set_title(graphname_[i]+"N")
        
    #     ax[j,k].grid()
    #     # print(i)
        

    # # ax[1,3].plot(baseframe['freq'],baseframe['aveamplitude'])
    # # ax[1,3].set_xlim([0, 200])
    # # ax[1,3].set_ylim([0, 2000])
    # # ax[1,3].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
    # # ax[1,3].set_ylabel("振幅スペクトル", size = 14, weight = "light",fontname="MS Gothic")
    # # ax[1,3].grid()
    # # ax[1,3].set_title("原音",fontname="MS Gothic")


    # # 透過率
    # fig, ax = plt.subplots(nrows=2,ncols=4) 
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # for i,x in enumerate(list_):
    #     if i==7: break
    #     j=i//4
    #     k=i%4
    #     # thrtf = x[x.freq<200]
    #     ax[j,k].plot(x['freq'],x['atten'])#, label=graphname_[i]+'N')# label='{}'.format(i)+'N')
    #     ax[j,k].set_xlim([1, 200])
    #     ax[j,k].set_ylim([-4000, 1000])
    #     ax[j,k].hlines(y=0,xmin=0,xmax=200,colors='red')
    #     # ax[j,k].set_ylim([1, 40])
    #     ax[j,k].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
    #     ax[j,k].set_ylabel("減衰", size = 14, weight = "light",fontname="MS Gothic")
    #     # ax[j,k].set_xscale("log")
    #     ax[j,k].set_title(graphname_[i]+"N")
    #     # if i == 11:
    #     #     ax[j,k].plot(baseframe['freq'],baseframe['atte'])
    #     #     ax[j,k].set_title("原音",fontname="MS Gothic")
    #     # # ax[j,k].grid()
    #     # plt.legend()
    #     # if i == 8:
    #     #     break
    #     # print(str(k) + ','+ str(j)  +'\n')
        

    # print(FFTsums)
    # print("最適荷重は:" + str(graphname_[FFTsums.index(max(FFTsums))]) + "N")
    
    # # plt.show() 
    
    # # targetN = [graphname_[FFTsums.index(max(FFTsums))]]
    
    # # "最適荷重での聴取を開始"
    # # handler(callback)
        
        
        
