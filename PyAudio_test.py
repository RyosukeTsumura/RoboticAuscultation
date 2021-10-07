import pyaudio
import time

class AudioFilter():
    def __init__(self):
        # オーディオに関する設定
        self.p = pyaudio.PyAudio()
        self.channels = 1 # マイクがモノラルの場合は1にしないといけない
        self.rate = 16000 # DVDレベルなので重かったら16000にする
        self.format = pyaudio.paInt16
        self.stream = self.p.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        output=True,
                        input=True,
                        stream_callback=self.callback)

    # コールバック関数（再生が必要なときに呼び出される）
    def callback(self, in_data, frame_count, time_info, status):
        out_data = in_data
        return (out_data, pyaudio.paContinue)

    def close(self):
        self.p.terminate()

if __name__ == "__main__":
    # AudioFilterのインスタンスを作る場所
    af = AudioFilter()

    # ストリーミングを始める場所
    af.stream.start_stream()

    # ノンブロッキングなので好きなことをしていていい場所
    while af.stream.is_active():
        time.sleep(0.1)

    # ストリーミングを止める場所
    af.stream.stop_stream()
    af.stream.close()
    af.close()

# import pyaudio
# import wave

# # FILE_PATH = "C:/Users/rtsumura1990/Documents/Program/RoboticAuscultation/test.wav"
# FILE_PATH = "test.wav"
# CHUNK = 1024
 
# wf = wave.open(FILE_PATH, 'rb')
 
# # instantiate PyAudio (1)
# p = pyaudio.PyAudio()
 
# # open stream (2)
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)
 
# # read data
# data = wf.readframes(CHUNK)
 
# # play stream (3)
# while len(data) > 0:
#     stream.write(data)
#     data = wf.readframes(CHUNK)
 
# # stop stream (4)
# stream.stop_stream()
# stream.close()
 
# # close PyAudio (5)
# p.terminate()
