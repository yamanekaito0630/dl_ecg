import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
import utils as u

# グラフのfont-familyの指定
plt.rcParams["font.family"] = "MS Gothic"
font_size = 20

dt = 0.01  # サンプリング間隔（時間）
t = np.arange(-2, 2, dt)
sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))


plt.plot(t, sig)
plt.title("波形", fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.xlabel("時間 [s]", fontsize=font_size)
plt.ylabel("周波数 [Hz]", fontsize=font_size)
plt.tight_layout()
plt.show()

wavelet_type = 'cmor1.5-1.0'
wav = pywt.ContinuousWavelet(wavelet_type)

# precisionによってデータ個数(len)が変わる
int_psi, x = pywt.integrate_wavelet(wav)

fs = 1 / dt  # サンプリング周波数
nq_f = fs / 2.0  # ナイキスト周波数

# 解析したい周波数のリスト（ナイキスト周波数以下）
# 1 Hz ～ nq_f Hzの間を等間隔に50分割
freqs = u.freq_logspace(1, 10, 5000)

# サンプリング周波数に対する比率を算出
freqs_rate = freqs / fs

# スケール：サンプリング周波数＝1:fs(1/dt)としてスケールに換算
scales = 1 / freqs_rate
# 逆順に入れ替え
scales = scales[::-1]

frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)

# スケール：サンプリング周波数＝1:fs(1/dt)として換算
frequencies = frequencies_rate / dt

cwtmatr, freqs_rate = pywt.cwt(sig, scales=scales, wavelet=wavelet_type)

x_positions, x_labels = u.get_ticks_label_set(t, num=4)
y_positions, y_labels = u.get_ticks_label_set(freqs[::-1], num=10)

n = 1  # 切り捨て桁数
y_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in y_labels]
print(y_labels)

n = 2  # 切り捨て桁数
x_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in x_labels]
print(x_labels)

plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='coolwarm')
plt.title("波形のウェーブレット変換図", fontsize=font_size)
plt.yticks(y_positions, y_labels, fontsize=font_size)
plt.xticks(x_positions, x_labels, fontsize=font_size)
plt.xlabel("時間 [s]", fontsize=font_size)
plt.ylabel("周波数 [Hz]", fontsize=font_size)
plt.colorbar().ax.tick_params(axis='y', right='off', labelsize=font_size)
plt.tight_layout()
plt.show()
