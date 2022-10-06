import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from matplotlib.colors import LogNorm
import utils as u
import wfdb

# グラフのfont-familyの指定
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 20
font_size = 20
data = '233'
arr_name = '心室性期外収縮'
arr_class = 'V'
person = 4

# 抽出時間
start = 900
end = 1000

record = wfdb.rdrecord('data/MIT-BIH/' + data, sampfrom=0, sampto=648000, physical=False, channels=[0, ])


# [mV]に変換する関数
def potential_conversion(val):
    return (val - record.baseline) / record.adc_gain


r = range(0, len(record.d_signal))
signals = map(potential_conversion, record.d_signal)
signals = list(signals)

# データ長
N = len(signals)

# サンプリング周期
dt = 1 / record.fs

x = np.linspace(0.0, N * dt, N)
y = signals

file_name = "text/potential/potential.txt"

with open(file_name, 'w') as f:
    for d in y:
        f.write("%s\n" % np.round(d, decimals=3))

with open(file_name, encoding="cp932") as f:
    data_lines = f.read()

# 文字列置換
data_lines = data_lines.replace('[', ']').replace(']', '')

# 同じファイル名で保存
with open(file_name, mode="w", encoding="cp932") as f:
    f.write(data_lines)

# サンプリング周波数
fs = 360

# サンプリング周期
dt = 1 / fs

# 生のテキストデータを取得
signal = np.loadtxt('text/potential/potential.txt')

# モルレーを指定
wavelet_type = 'cmor1.5-1.0'
wav = pywt.ContinuousWavelet(wavelet_type)

# precisionによってデータ個数(len)が変わる
int_psi, x = pywt.integrate_wavelet(wav)

# 解析したい周波数のリスト（ナイキスト周波数以下）
# 1 Hz ～ nq_f Hzの間を等間隔に1000分割
freqs = u.freq_logspace(0.1, 20, 100)

# サンプリング周波数に対する比率を算出
freqs_rate = freqs / fs

# スケール：サンプリング周波数＝1:fs(1/dt)としてスケールに換算
scales = 1 / freqs_rate

# 逆順に入れ替え
scales = scales[::-1]

frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)
frequencies = frequencies_rate / dt

# 800枚の画像を保存
for num in range(600):
    t = np.linspace(start, end, (end - start) * fs)
    sig = signal[start * fs:end * fs]

    start += 1
    end += 1

    mean = np.mean(sig)
    std = np.std(sig)


    def norm(val):
        return (val - mean) / std


    sig = map(norm, sig)
    sig = list(sig)

    cwtmatr, _ = pywt.cwt(sig, scales=scales, wavelet=wavelet_type)

    # x軸、y軸のラベル表示設定
    x_positions, x_labels = u.get_ticks_label_set(t, num=4)
    y_positions, y_labels = u.get_ticks_label_set(freqs[::-1], num=10)

    # 切り捨て桁数
    n = 2
    y_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in y_labels]

    # 切り捨て桁数
    n = 0
    x_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in x_labels]

    # f_min = int(np.min(np.abs(cwtmatr)) * 10) / 10 if int(np.min(np.abs(cwtmatr)) * 10) / 10 != 0.0 else 0.1
    f_min = 0.1
    f_max = np.max(np.abs(cwtmatr))

    fig = plt.figure(figsize=(3.12, 3.12))
    plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='coolwarm', norm=LogNorm(vmin=f_min, vmax=f_max))
    plt.yticks(y_positions, y_labels, fontsize=font_size)
    plt.xticks(x_positions, x_labels, fontsize=font_size)
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 画像データ保存
    if person == 1:
        plt.savefig('./img/' + arr_class + '/' + arr_class + '(' + str(num + 1) + ')')
    elif person == 2:
        plt.savefig('./img/' + arr_class + '/' + arr_class + '(' + str(num + 601) + ')')
    elif person == 3:
        plt.savefig('./img/' + arr_class + '/' + arr_class + '(' + str(num + 1201) + ')')
    else:
        plt.savefig('./img/' + arr_class + '/' + arr_class + '(' + str(num + 1801) + ')')
    plt.close()
