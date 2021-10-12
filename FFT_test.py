import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

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

audio = pyaudio.PyAudio() # create pyaudio instantiation

# ストリームの作成
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)

# x軸
wave_x = np.linspace(0, samp_rate, chunk)
chunk2 = int(chunk/2)
wave_x2 = wave_x[0:chunk2]

print("recording")

# 音声の取得とFFT処理
while True:
    try:
        # 音声データの取得
        data = stream.read(chunk)
        ndarray = np.frombuffer(data, dtype='int16')
        
        # FFT
        wave_y = np.fft.fft(ndarray)
        wave_y = np.abs(wave_y)

        # dB変換
        wave_y = db(np.sqrt(wave_y),dBref)

        # データ整理
        wave_y2 = wave_y[0:chunk2]
        
        # ピーク検出処理
        peak = max(wave_y2)
        print('ピークdB:',peak)
        print('ピーク周波数：', wave_x2[np.argmax(wave_y2)])

        # グラフ表示
        plt.plot(wave_x2,wave_y2)
        plt.draw()
        plt.pause(sleepTime)
        plt.cla()
    except KeyboardInterrupt:
        print("Ctrl+Cで停止しました")
        break

print("finished recording")

# ストリームの終了
stream.stop_stream()
stream.close()
audio.terminate()