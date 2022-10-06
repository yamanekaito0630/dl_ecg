import math

import matplotlib.pyplot as plt
import numpy as np
import pywt
import wfdb
from matplotlib.colors import LogNorm
from numpy.fft import fft, fftfreq


class FurieAndWavletCreator(object):
    # 変数初期化
    def __init__(self, data, arr_name, start, end):
        self.data = data
        self.start = start
        self.end = end
        self.arr_name = arr_name
        self.font_size = 20
        self.fs = 360
        self.cal1, self.cal2, self.record, self.N, self.dt, self.t, self.y, self.freq, self.F, self.sig = np.full(10,
                                                                                                                  False)

    @staticmethod
    def read_lines_file(file_name):
        with open(file_name, 'r') as file:
            return file.readlines()

    @staticmethod
    def save_file(file_name, text):
        with open(file_name, 'w') as file:
            file.write(text)

    @staticmethod
    def freq_logspace(start_f: float, stop_f: float, num: int):
        log_base = 10
        low_mul = np.log10(start_f)
        high_mul = np.log10(stop_f)
        return np.logspace(low_mul, high_mul, num=num, base=log_base)

    @staticmethod
    def get_ticks_label_set(all_labels, num: int):
        length = len(all_labels)
        step = length // (num - 1)
        pick_positions = np.arange(0, length, step)
        pick_labels = all_labels[::step]
        return pick_positions, pick_labels

    # 生のデータから[mV]へ変換
    def potential_conversion(self, val):
        return (val - self.record.baseline) / self.record.adc_gain

    # 時系列データのテキストファイルを保存
    def save_ecg_text(self):
        self.record = wfdb.rdrecord('data/MIT-BIH/' + self.data, sampfrom=0, sampto=648000, physical=False,
                                    channels=[0, ])
        signals = map(self.potential_conversion, self.record.d_signal)
        signals = list(signals)
        self.y = signals
        file_name = "text/potential/potential.txt"

        with open(file_name, 'w') as f:
            for d in self.y:
                f.write("%s\n" % np.round(d, decimals=3))

        with open(file_name, encoding="cp932") as f:
            data_lines = f.read()

        data_lines = data_lines.replace('[', ']').replace(']', '')

        with open(file_name, mode="w", encoding="cp932") as f:
            f.write(data_lines)

    # 時系列データとフーリエ変換図を表示
    def show_furie(self):
        # 生のテキストデータを取得
        signal = np.loadtxt('text/potential/potential.txt')
        self.t = np.linspace(self.start, self.end, (self.end - self.start) * self.fs)
        self.sig = signal[self.start * self.fs:self.end * self.fs]

        # 波形の表示
        plt.rcParams["font.family"] = "MS Gothic"
        plt.figure(figsize=(8, 4))
        plt.plot(self.t, self.sig, '-', lw=1)
        plt.title(self.arr_name + "(" + self.data + ')', fontsize=self.font_size)
        plt.xlabel('測定時間 [s]', fontsize=self.font_size)
        plt.ylabel('振幅 [mV]', fontsize=self.font_size)
        plt.xlim(self.start, self.end)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.tight_layout()
        plt.show()

        self.N = len(self.sig)
        self.dt = 1 / self.record.fs
        self.freq = fftfreq(self.N, self.dt)
        self.F = np.abs(fft(self.sig))
        nyquist = 1 / self.dt / 2.0
        self.F = self.F / (self.N / 2)
        self.F[(self.freq > nyquist)] = 0

        # 振幅スペクトル
        plt.rcParams["font.family"] = "MS Gothic"
        plt.figure(figsize=(8, 4))
        plt.plot(self.freq[1:self.N // 2], np.abs(self.F)[1:self.N // 2], lw=1)
        plt.title(self.arr_name + "(" + self.data + ")のフーリエ変換図", fontsize=self.font_size)
        plt.xlabel('周波数 [Hz]', fontsize=self.font_size)
        plt.ylabel('振幅 [mV]', fontsize=self.font_size)
        plt.xlim(0, 10)
        plt.ylim(0, )
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.tight_layout()
        plt.savefig(self.data + "_furie.png")
        plt.show()

    # フーリエ変換のテキストファイルを保存
    def save_furie_text(self):
        with open('text/amp_freq/freq.txt', 'w') as f:
            for d in self.freq[1:self.N // 2]:
                f.write("%s\n" % round(d, 1))

        with open('text/amp_freq/amp.txt', 'w') as f:
            for d in np.abs(self.F)[1:self.N // 2]:
                f.write("%s\n" % round(d, 4))

        self.cal1 = self.read_lines_file('text/amp_freq/freq.txt')
        self.cal2 = self.read_lines_file('text/amp_freq/amp.txt')
        self.cal1 = list(map(lambda x: x.strip(), self.cal1))
        self.cal2 = list(map(lambda x: x.strip(), self.cal2))
        lines = ["{0},{1}\n".format(line1, line2) for line1, line2 in zip(self.cal1, self.cal2)]
        self.save_file('mitbih_furie/furie_' + self.data + '.txt', "".join(lines))

    def show_norm_wavlet(self):
        # モルレーを指定
        wavelet_type = 'cmor1.5-1.0'
        freqs = self.freq_logspace(0.1, 20, 100)
        freqs_rate = freqs / self.fs
        scales = 1 / freqs_rate
        scales = scales[::-1]
        mean = np.mean(self.sig)
        std = np.std(self.sig)

        def norm(val):
            return (val - mean) / std

        sig = map(norm, self.sig)
        sig = list(sig)

        cwtmatr, freqs_rate = pywt.cwt(sig, scales=scales, wavelet=wavelet_type)
        x_positions, x_labels = self.get_ticks_label_set(self.t, num=4)
        y_positions, y_labels = self.get_ticks_label_set(freqs[::-1], num=10)

        n = 2
        y_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in y_labels]

        n = 0
        x_labels = [math.floor(d * 10 ** n) / (10 ** n) for d in x_labels]

        f_min = int(np.min(np.abs(cwtmatr)) * 10) / 10 if int(np.min(np.abs(cwtmatr)) * 10) / 10 != 0.0 else 0.1
        f_max = np.max(np.abs(cwtmatr))

        plt.rcParams["font.family"] = "MS Gothic"
        plt.figure(figsize=(8, 4))
        plt.imshow(np.abs(cwtmatr), aspect='auto', cmap='coolwarm', norm=LogNorm(vmin=f_min, vmax=f_max))
        plt.yticks(y_positions, y_labels, fontsize=self.font_size)
        plt.xticks(x_positions, x_labels, fontsize=self.font_size)
        plt.title(self.arr_name + "(" + self.data + ")のウェーブレット変換図", fontsize=self.font_size)
        plt.xlabel("時間[s]", fontsize=self.font_size)
        plt.ylabel("周波数[Hz]", fontsize=self.font_size)
        plt.colorbar(label='電圧 [mV]').ax.tick_params(axis='y', right='off', labelsize=self.font_size)
        plt.tight_layout()
        plt.savefig(self.data + "_wavlet.png")
        plt.show()

    # 実行
    def run(self):
        self.save_ecg_text()
        self.show_furie()
        self.show_norm_wavlet()
        self.save_furie_text()


DATA = '200'
ARR_NAME = '正常'
START = 300
END = 310

FurieAndWavletCreator(DATA, ARR_NAME, START, END).run()
