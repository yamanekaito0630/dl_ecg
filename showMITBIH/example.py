import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import utils as u

# グラフのfont-familyの指定
plt.rcParams["font.family"] = "MS Gothic"
font_size = 30

# データのパラメータ
N = 1000  # サンプル数
dt = 0.01  # サンプリング間隔
A1, A2 = 3, 5
f1, f2 = 2, 7  # 周波数
t = np.arange(0, N * dt, dt)  # 時間軸
freq = np.linspace(0, 1.0 / dt, N)  # 周波数軸

# 信号を生成（周波数f1の正弦波+周波数f2の正弦波）
y = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# 波形
plt.plot(t, y)
plt.title('波形', fontsize=font_size)
plt.xlim(0, 1)
plt.xlabel("秒 [s]", fontsize=font_size)
plt.ylabel("シグナル", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.tight_layout()
plt.show()

# 波形の表示結果をテキストファイルに保存
with open('example_wave/time.txt', 'w') as f:
    for d in t:
        f.write("%s\n" % round(d, 3))

with open('example_wave/potential.txt', 'w') as f:
    for d in y:
        f.write("%s\n" % round(d, 4))

# 読み込んだファイルをlist型で受け取る
cal1 = u.readlines_file('example_wave/time.txt')
cal2 = u.readlines_file('example_wave/potential.txt')

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

u.save_file('example_wave/example_wave.txt', "".join(lines))

# 高速フーリエ変換(FFT)
F = np.fft.fft(y)
# 振幅スペクトルを計算
amplitude = np.abs(F)

# 調整
F_amplitude = amplitude / N * 2
F_amplitude[0] = F_amplitude[0] / 2

# フーリエ変換図
plt.plot(freq[:int(N / 2) + 1], F_amplitude[:int(N / 2) + 1])
plt.title('フーリエ変換図', fontsize=font_size)
plt.xlabel("周波数 [Hz]", fontsize=font_size)
plt.ylabel("振幅", fontsize=font_size)
plt.xlim(0, (1 / dt) / 10)
plt.ylim(0, )
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.tight_layout()
plt.show()

# フーリエ変換の結果をテキストファイルに保存
with open('example_furie/freq.txt', 'w') as f:
    for d in freq[1:N // 2]:
        f.write("%s\n" % round(d, 1))

with open('example_furie/amp.txt', 'w') as f:
    for d in F_amplitude[1:N // 2]:
        f.write("%s\n" % round(d, 4))

# 読み込んだファイルをlist型で受け取る
cal1 = u.readlines_file('freq.txt')
cal2 = u.readlines_file('amp.txt')

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

u.save_file('example_furie/example_furie.txt', "".join(lines))
