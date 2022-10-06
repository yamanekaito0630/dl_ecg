import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from matplotlib.colors import LogNorm

# グラフのfont-familyの指定
from numpy.fft import fftfreq, fft

plt.rcParams["font.family"] = "MS Gothic"
font_size = 25
plt.rcParams["font.size"] = font_size


# 時間軸を生成する関数
def unit_conversion(val, f=100):
    return val / f


# 生のデータを[mV]に変換する関数
def potential_conversion(val, n=10, vcc=3.3, gain=1100):
    return ((((val / 2 ** n) - 0.5) * vcc) / gain) * 1000


def get_ticks_label_set(all_labels, num: int):
    length = len(all_labels)
    step = length // (num - 1)
    pick_positions = np.arange(0, length, step)
    pick_labels = all_labels[::step]
    return pick_positions, pick_labels


def readlines_file(file_name):
    # 行毎のリストを返す
    with open(file_name, 'r') as file:
        return file.readlines()


def save_file(file_name, text):
    with open(file_name, 'w') as file:
        file.write(text)


# 対数表記する関数
def freq_logspace(start_f: float, stop_f: float, num: int):
    """
    numで指定した個数のデータをlogspaceで取得する
    取得する値の範囲をstart_f, stop_fで指定
    """
    log_base = 2
    low_mul = np.log2(start_f)
    high_mul = np.log2(stop_f)
    return np.logspace(low_mul, high_mul, num=num, base=log_base)


# 自分のECGを表示する関数
def show_my_ecg(signal, move_name, start, end):
    signals = map(potential_conversion, signal)
    signals = list(signals)

    # サンプリング周波数
    fs = 100

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


# 自分のECGをフーリエ変換する関数
def show_my_furie(signal, move_name, start, end):
    # サンプリング周波数
    fs = 100

    # サンプリング周期
    dt = 1 / fs

    # 生のデータを[mV]に変換
    ecg_signal = list(map(potential_conversion, signal))

    t = np.linspace(start, end, (end - start) * fs)
    y = ecg_signal[start * fs:end * fs]

    # データ長
    N = len(y)

    # 周波数スケール
    freq = fftfreq(N, dt)

    F = np.abs(fft(y))

    # 正規化
    F = F / (N / 2)

    # 振幅スペクトル
    plt.figure(figsize=(8, 4))
    plt.plot(freq[1:N // 2], np.abs(F)[1:N // 2])
    plt.title(move_name + '時のフーリエ変換図', fontsize=font_size)
    plt.xlabel('周波数 [Hz]', fontsize=font_size)
    plt.ylabel('振幅 [mV]', fontsize=font_size)
    plt.xlim(0, 10)
    plt.ylim(0, )
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()

    with open('log/marge/time.txt', 'w') as f:
        for d in t:
            f.write("%s\n" % round(d, 3))

    with open('log/marge/potential.txt', 'w') as f:
        for d in y:
            f.write("%s\n" % round(d, 4))

    # 読み込んだファイルをlist型で受け取る
    cal1 = readlines_file('log/marge/time.txt')
    cal2 = readlines_file('log/marge/potential.txt')

    # 改行や空白文字を削除
    cal1 = list(map(lambda x: x.strip(), cal1))
    cal2 = list(map(lambda x: x.strip(), cal2))

    # タブ区切りで並べたリストを作成
    lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

    save_file('log/ecg_wave.txt', "".join(lines))

    with open('log/marge/freq.txt', 'w') as f:
        for d in freq[0:N // 2]:
            f.write("%s\n" % round(d, 1))

    with open('log/marge/amp.txt', 'w') as f:
        for d in np.abs(F)[0:N // 2]:
            f.write("%s\n" % round(d, 4))

    # 読み込んだファイルをlist型で受け取る
    cal1 = readlines_file('log/marge/freq.txt')
    cal2 = readlines_file('log/marge/amp.txt')

    # 改行や空白文字を削除
    cal1 = list(map(lambda x: x.strip(), cal1))
    cal2 = list(map(lambda x: x.strip(), cal2))

    # タブ区切りで並べたリストを作成
    lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

    save_file('log/ecg_furie.txt', "".join(lines))


# 自分のECGをウェーブレット変換する関数
def show_my_wavelet(signal, move_name, start, end):
    signals = map(potential_conversion, signal)
    signals = list(signals)

    # サンプリング周波数
    fs = 100

    t = np.linspace(start, end, (end - start) * fs)
    sig = signals[start * fs:end * fs]

    # モルレーを指定
    wavelet_type = 'cmor1.5-1.0'
    wav = pywt.ContinuousWavelet(wavelet_type)

    # precisionによってデータ個数(len)が変わる
    int_psi, _ = pywt.integrate_wavelet(wav)

    # 解析したい周波数のリスト（ナイキスト周波数以下）
    # 1 Hz ～ nq_f Hzの間を等間隔に1000分割
    freqs = freq_logspace(0.1, 20, 100)

    # サンプリング周波数に対する比率を算出
    freqs_rate = freqs / fs

    # スケール：サンプリング周波数＝1:fs(1/dt)としてスケールに換算
    scales = 1 / freqs_rate

    # 逆順に入れ替え
    scales = scales[::-1]

    mean = np.mean(sig)
    std = np.std(sig)

    # 標準化する関数
    def norm(val):
        return (val - mean) / std

    sig = map(norm, sig)
    sig = list(sig)

    cwtmatr, _ = pywt.cwt(sig, scales=scales, wavelet=wavelet_type)

    # x軸、y軸のラベル表示設定
    x_positions, x_labels = get_ticks_label_set(t, num=4)
    y_positions, y_labels = get_ticks_label_set(freqs[::-1], num=5)

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

    print('最高電圧：' + str(f_max))

    # グラフ表示
    plt.figure(figsize=(8, 4))
    plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='coolwarm', norm=LogNorm(vmin=f_min, vmax=f_max))
    plt.yticks(y_positions, y_labels, fontsize=font_size)
    plt.xticks(x_positions, x_labels, fontsize=font_size)
    plt.title(move_name + '時のウェーブレット変換図', fontsize=font_size)
    plt.xlabel("時間[s]", fontsize=font_size)
    plt.ylabel("周波数[Hz]", fontsize=font_size)
    plt.colorbar(label='電圧 [mV]').ax.tick_params(axis='y', right='off', labelsize=font_size)
    plt.tight_layout()
    plt.show()
