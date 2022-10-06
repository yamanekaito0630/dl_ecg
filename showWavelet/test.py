import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from matplotlib.colors import LogNorm
import utils as u

# グラフのfont-familyの指定
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 20
font_size = 20
move_name = 'スクワット'

signal = np.loadtxt('beat/squat2.txt', comments='#')[:, 5]

r = range(0, len(signal))
signals = map(u.potential_conversion, signal)
signals = list(signals)

# データ長
N = len(signals)

# サンプリング周波数
fs = 100

# サンプリング周期
dt = 1 / fs

x = np.linspace(0.0, N * dt, N)
y = signals

# 抽出時間
start = 10
end = 20

t = np.linspace(start, end, (end - start) * fs)
sig = signals[start * fs:end * fs]

# 波形の表示
plt.figure(figsize=(8, 4))
plt.plot(t, sig, '-', lw=1)
plt.title(move_name + '時の波形', fontsize=font_size)
plt.xlabel('測定時間 [s]', fontsize=font_size)
plt.ylabel('振幅 [mV]', fontsize=font_size)
plt.xlim(start, end)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.tight_layout()
plt.show()

# モルレーを指定
wavelet_type = 'cmor1.5-1.0'
wav = pywt.ContinuousWavelet(wavelet_type)

# precisionによってデータ個数(len)が変わる
int_psi, _ = pywt.integrate_wavelet(wav)

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

mean = np.mean(sig)
std = np.std(sig)


# [mV]に変換する関数
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
print(y_labels)

# 切り捨て桁数
n = 0
x_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in x_labels]
print(x_labels)

f_min = int(np.min(np.abs(cwtmatr)) * 10) / 10 if int(np.min(np.abs(cwtmatr)) * 10) / 10 != 0.0 else 0.1
f_max = np.max(np.abs(cwtmatr))

print(f_max)

plt.figure(figsize=(8, 4))
plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='coolwarm', norm=LogNorm(vmin=f_min, vmax=f_max))
plt.yticks(y_positions, y_labels, fontsize=font_size)
plt.xticks(x_positions, x_labels, fontsize=font_size)
plt.title(move_name + '時のウェーブレット変換図', fontsize=font_size)
plt.xlabel("時間[s]", fontsize=font_size)
plt.ylabel("周波数[Hz]", fontsize=font_size)
plt.colorbar(label='電圧 [mV]').ax.tick_params(axis='y', right='off', labelsize=font_size)
# plt.clim(0, 10)
plt.tight_layout()
plt.show()
