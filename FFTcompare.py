import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import numpy
import re
from scipy import optimize as opt

from natsort import natsorted



basefile = './data/test/FFT/0.5N_FFT.csv'
file = './data/test/FFT/'
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

FFTsums = []

# 羅列
fig, ax = plt.subplots(nrows=2,ncols=4) 
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for i,x in enumerate(list_):
    if i==7: break
    j=i//4
    k=i%4
    # thrtf = x[x.freq<400]


    ax[j,k].plot(x['freq'],x['aveamplitude'])#, label=graphname_[i]+'N')# label='{}'.format(i)+'N')
    # ax[j,k].plot(baseframe['freq'],baseframe['aveamplitude'])
    goukei = x['aveamplitude'].sum()
    goukei = round(goukei)
    FFTsums.append(goukei)
    # ax[j,k].text(50,1934,"振幅スペクトルの合計" + str(goukei) ,backgroundcolor="#ffffff",fontname="MS Gothic")
    print(goukei)
    gensui = (1-x['aveamplitude'].sum() / baseframe['aveamplitude'].sum())*100
    gensui = round(gensui)
    # ax[j,k].text(130,1734,"減衰率:" + str(gensui) + "%",backgroundcolor="#ffffff",fontname="MS Gothic")
    ax[j,k].text(190,1934,"振幅スペクトルの合計:" + str(goukei) + "\n減衰率:" + str(gensui) + "%",backgroundcolor="#ffffff",fontname="MS Gothic", horizontalalignment="right")


    ax[j,k].set_xlim([0, 200])
    ax[j,k].set_ylim([0, 2500])
    # ax[j,k].set_ylim([1, 40])
    ax[j,k].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
    ax[j,k].set_ylabel("振幅スペクトル", size = 14, weight = "light",fontname="MS Gothic")
    # ax[j,k].set_yscale("log")
    ax[j,k].set_title(graphname_[i]+"N")
    
    ax[j,k].grid()
    # print(i)
    

# ax[1,3].plot(baseframe['freq'],baseframe['aveamplitude'])
# ax[1,3].set_xlim([0, 200])
# ax[1,3].set_ylim([0, 2000])
# ax[1,3].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
# ax[1,3].set_ylabel("振幅スペクトル", size = 14, weight = "light",fontname="MS Gothic")
# ax[1,3].grid()
# ax[1,3].set_title("原音",fontname="MS Gothic")


# 透過率
fig, ax = plt.subplots(nrows=2,ncols=4) 
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for i,x in enumerate(list_):
    if i==7: break
    j=i//4
    k=i%4
    # thrtf = x[x.freq<200]
    ax[j,k].plot(x['freq'],x['atten'])#, label=graphname_[i]+'N')# label='{}'.format(i)+'N')
    ax[j,k].set_xlim([1, 200])
    ax[j,k].set_ylim([-4000, 1000])
    ax[j,k].hlines(y=0,xmin=0,xmax=200,colors='red')
    # ax[j,k].set_ylim([1, 40])
    ax[j,k].set_xlabel("周波数[Hz]", size = 14, weight = "light",fontname="MS Gothic")
    ax[j,k].set_ylabel("減衰", size = 14, weight = "light",fontname="MS Gothic")
    # ax[j,k].set_xscale("log")
    ax[j,k].set_title(graphname_[i]+"N")
    # if i == 11:
    #     ax[j,k].plot(baseframe['freq'],baseframe['atte'])
    #     ax[j,k].set_title("原音",fontname="MS Gothic")
    # # ax[j,k].grid()
    # plt.legend()
    # if i == 8:
    #     break
    # print(str(k) + ','+ str(j)  +'\n')
    



print(FFTsums)
print("最適荷重は:" + str(graphname_[FFTsums.index(max(FFTsums))]) + "N")
plt.show() 

