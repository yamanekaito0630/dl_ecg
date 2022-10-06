import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
import utils as u

# グラフのfont-familyの指定
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["font.size"] = 20
font_size = 20
move_name = 'スクワット'

# 抽出時間
start = 10
end = 20

# サンプリング周波数
fs = 100

# サンプリング周期
dt = 1 / fs

# ナイキスト周波数
nyquist = 1 / dt / 2.0

# 生のテキストデータを取得
signal = np.loadtxt('beat/squat2.txt', comments='#')[:, 5]

# 生のデータを[mV]に変換
ecg_signal = list(map(u.potential_conversion, signal))

t = np.linspace(start, end, (end - start) * fs)
y = ecg_signal[start * fs:end * fs]

# データ長
N = len(y)

# 周波数スケール
freq = fftfreq(N, dt)

F = np.abs(fft(y))

# 正規化
F = F / (N / 2)

# ローパスフィルタ
F[(freq > nyquist)] = 0

# # 信号データ
# plt.figure(figsize=(8, 4))
# plt.plot(t, y)
# plt.title(move_name+'時の波形', fontsize=font_size)
# plt.xlabel('測定時間 [s]', fontsize=font_size)
# plt.ylabel('振幅 [mV]', fontsize=font_size)
# plt.xlim(start, end)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.tight_layout()
# plt.show()

# 振幅スペクトル
plt.figure(figsize=(8, 4))
plt.plot(freq[1:N // 2], np.abs(F)[1:N // 2])
plt.title(move_name+'時のフーリエ変換図', fontsize=font_size)
plt.xlabel('周波数 [Hz]', fontsize=font_size)
plt.ylabel('振幅 [mV]', fontsize=font_size)
plt.xlim(0, 10)
plt.ylim(0, )
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.tight_layout()
plt.show()

with open('time.txt', 'w') as f:
    for d in t:
        f.write("%s\n" % round(d, 3))

with open('potential.txt', 'w') as f:
    for d in y:
        f.write("%s\n" % round(d, 4))

# 読み込んだファイルをlist型で受け取る
cal1 = u.readlines_file('time.txt')
cal2 = u.readlines_file('potential.txt')

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

u.save_file('ecg_wave.txt', "".join(lines))

with open('freq.txt', 'w') as f:
    for d in freq[0:N // 2]:
        f.write("%s\n" % round(d, 1))

with open('amp.txt', 'w') as f:
    for d in np.abs(F)[0:N // 2]:
        f.write("%s\n" % round(d, 4))

# 読み込んだファイルをlist型で受け取る
cal1 = u.readlines_file('freq.txt')
cal2 = u.readlines_file('amp.txt')

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

u.save_file('ecg_furie.txt', "".join(lines))
