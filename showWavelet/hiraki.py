# ライブラリのインポート
import functools
import glob
import h5py
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import numpy as np
import os
import pycwt as wavelet
import pywt
import random
import shutil
import time


# MITのデータを読み込みウェーブレット変換図を作成するクラス
class BaseECGDatasetPreprocessor(object):
    # 初期化関数
    # dataset_path: MITデータセットのファイルパス
    # save_foledr_path: 作成したウェーブレット変換図を保存するフォルダパス
    # window_sec: 何秒の心電データを1枚のウェーブレット変換図にするのか[秒]
    # freq_range: 何ヘルツから15[Hz]までの範囲をウェーブレット変換図にするのか
    # sample_rate: サンプリングレート
    def __init__(
            self,
            dataset_path,
            save_folder_path,
            window_sec=100,
            freq_range=0.1,
            sample_rate=360.
    ):
        self.save_folder_path = save_folder_path
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.window_size = window_sec * sample_rate
        self.freq_range = freq_range
        self.dt = 1 / sample_rate
        # MITのラベルをどのように扱うのかのリスト
        # invalid_symbolsのリストは使用しないラベルの心電データ．
        # regular_symbolsのリストは正常ラベルの心電データ
        # irregular_symbolsのリストは異常ラベルの心電データ
        self.invalid_symbols = ['Q', '?', 'A', 'R', 'e', 'j', '+', '~', '|', 'x', 'J', '"', 'E', 'F', 'a', '!', '[',
                                ']', 'S']
        self.regular_symbols = ['N']
        self.irregular_symbols = ['L', 'V']
        self.normal_num = 0
        self.abnormal1_num = 0
        self.abnormal2_num = 0

        self.train_normal_num = 0
        self.train_abnormal1_num = 0
        self.train_abnormal2_num = 0
        self.test_normal_num = 0
        self.test_abnormal1_num = 0
        self.test_abnormal2_num = 0
        self.name_tag = 1

    def load_data(
            self,
            base_record,
            channel=0  # [0, 1]
    ):
        record_name = os.path.join(self.dataset_path, str(base_record))
        # 心電データを読み込み
        signals, fields = wfdb.rdsamp(record_name)
        assert fields['fs'] == self.sample_rate
        # ラベルを読み込み
        annotation = wfdb.rdann(record_name, 'atr')
        symbols = annotation.symbol
        positions = annotation.sample
        return np.array(signals[:, channel]), symbols, positions

    # 心電データを標準化もしくは正規化を行う関数"minmax"が正規化，"std"が標準化を行う．
    def normalize_signal(
            self,
            signal,
            method='std'
    ):
        if method == 'minmax':
            # Min-Max scaling
            min_val = np.min(signal)
            max_val = np.max(signal)
            return (signal - min_val) / (max_val - min_val)
        elif method == 'std':
            # Zero mean and unit variance
            signal = (signal - np.mean(signal)) / np.std(signal)
            return signal
        else:
            raise ValueError("Invalid method: {}".format(method))

    # 一枚のウェーブレット変換図にする時間ごとに心電データを抽出して，ラベル付けを行う関数
    # signal: 使用する心電データの配列
    # symbol: ラベルの配列
    # position: ラベルが心電データのどの範囲を指しているのかを格納したリスト
    # test_rate:全体の何割をテストデータとして用いるのかの指定
    def segment_data(self, signal, symbols, positions, test_rate=0.3):
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        i = 0

        std_line = int(len(signal) * (1.0 - test_rate))
        while True:
            is_normal = True
            is_abnormal1 = False
            is_abnormal2 = False

            is_valid = True
            start = int(i)
            end = int(i + self.window_size)
            # 抽出した心電データがtarin用なのかtest用なのかの判定
            if end <= std_line:
                mode = 'train'
            elif end >= std_line and start <= std_line:
                is_valid = False
            elif start >= std_line:
                mode = 'test'

            # 心電データを指定した時間範囲に抽出
            segment = signal[start:end]
            if len(segment) != self.window_size:
                break

            # 抽出した心電データが有効かどうかとラベルが何なのかを判定
            for label_po, label in zip(positions, symbols):
                if label_po < start:
                    continue
                elif label_po > end:
                    break
                elif label in self.invalid_symbols:
                    is_valid = False
                    break
                elif label in self.regular_symbols:
                    continue
                elif label in ['V']:
                    is_normal = False
                    if is_abnormal2 == True:
                        is_valid = True
                        break
                    elif is_abnormal1 == False:
                        is_abnormal1 = True
                        continue
                elif label in ['L']:
                    is_normal = False
                    if is_abnormal1 == True:
                        is_valid = True
                        break
                    elif is_abnormal2 == False:
                        is_abnormal2 = True
                        continue
                else:
                    pass

            # 有効ならば，心電データとその心電データのラベルを配列に格納
            if is_valid:
                if mode == 'train':
                    train_X.append(segment)
                    if is_normal:
                        train_y.append(0)
                        self.normal_num = self.normal_num + 1
                        self.train_normal_num = self.train_normal_num + 1
                    elif is_abnormal1:
                        train_y.append(1)
                        self.abnormal1_num = self.abnormal1_num + 1
                        self.train_abnormal1_num = self.train_abnormal1_num + 1
                    else:
                        train_y.append(2)
                        self.abnormal2_num = self.abnormal2_num + 1
                        self.train_abnormal2_num = self.train_abnormal2_num + 1
                else:
                    test_X.append(segment)
                    if is_normal:
                        test_y.append(0)
                        self.normal_num = self.normal_num + 1
                        self.test_normal_num = self.test_normal_num + 1
                    elif is_abnormal1:
                        test_y.append(1)
                        self.abnormal1_num = self.abnormal1_num + 1
                        self.test_abnormal1_num = self.test_abnormal1_num + 1
                    else:
                        test_y.append(2)
                        self.abnormal2_num = self.abnormal2_num + 1
                        self.test_abnormal2_num = self.test_abnormal2_num + 1

            if is_valid == False:
                i = i + self.sample_rate // 6
            elif is_normal == True:
                i = i + int(self.sample_rate) * 3
            elif is_abnormal1:
                i = i + self.sample_rate * 3
            else:
                i = i + self.sample_rate // 2.5

            return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


# 心電データをウェーブレット変換する関数
# X: 心電データの配列
# Y: 心電データのラベルの配列
# mode: "train"か"test"用かの指定
def ecg_to_WT(self, X, Y, mode):
    # マザーウェーブレットの変数ω0
    omega0 = 6
    # マザーウェーブレットを定義
    mother = wavelet.Morlet(omega0)
    os.chdir(self.save_folder_path)
    parent_file_name = "range_" + str(self.freq_range)

    # ウェーブレット変換図を保存するフォルダがないなら作成
    if not os.path.isdir(os.path.join(parent_file_name, "regular", mode)):
        # shutil.rmtree(os.path.join(parent_file_name,label_name,wave_info[1],mode))
        os.makedirs(os.path.join(parent_file_name, "regular", mode))
    if not os.path.isdir(os.path.join(parent_file_name, "irregular_V", mode)):
        # shutil.rmtree(os.path.join(parent_file_name,label_name,wave_info[1],mode))
        os.makedirs(os.path.join(parent_file_name, "irregular_V", mode))
    if not os.path.isdir(os.path.join(parent_file_name, "irregular_L", mode)):
        # shutil.rmtree(os.path.join(parent_file_name,label_name,wave_info[1],mode))
        os.makedirs(os.path.join(parent_file_name, "irregular_L", mode))

    os.chdir(parent_file_name)

    for data, label in zip(X, Y):
        # 周波数範囲を指定してウェーブレット変換を行う
        dj = 0.1
        s0 = 0.06453422061295941  # 15Hzまで
        J = 86.9  # (np.log2(len(data) * self.dt / s0)) / dj #0.1Hz
        self.name_tag = self.name_tag + 1
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data, self.dt, dj, s0, J, mother)

        if label == 0:
            save_path = './regular/' + mode
        elif label == 1:
            save_path = './irregular_V/' + mode
        else:
            save_path = './irregular_L/' + mode
            # ウェーブレット変換した心電データをウェーブレット変換図として保存
        self._save_color_WT(np.abs(wave), save_path)

# クラスの全ての処理を順番に実行する関数
# train_record_list:train用に使用する心電データのファイルパス
# test_record_list:test用に使用する心電データのファイルパス
# normalize:心電データを標準化するかどうか
    # save_fig: 作成したウェーブレット変換図を保存するかどうか


def preprocess_dataset(self, train_record_list, test_record_list, normalize=True, save_fig=True):
    self.dic = {"N": 0}
    # preprocess training dataset
    self._preprocess_dataset_core(train_record_list, "train", normalize)
    # preprocess test dataset
    self._preprocess_dataset_core(test_record_list, "test", normalize)

    # ウェーブレット変換図を作成して保存する関数
    # WT_data: ウェーブレット変換した心電データを格納した配列
    # save_path: ウェーブレット変換図を保存するファイルパス


def save_color_WT(self, WT_data, save_path):
    f_min = int(np.min(WT_data) * 10) / 10 if int(np.min(WT_data) * 10) / 10 != 0.0 else 0.1
    f_max = np.max(WT_data)
    fig = plt.figure(figsize=(3.12, 3.12))
    mappable0 = plt.pcolormesh(np.flipud(WT_data), cmap='coolwarm', norm=LogNorm(vmin=f_min, vmax=f_max))
    plt.axis("off")
    img_file_name = "./img_" + str(self.name_tag)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.chdir(save_path)
    # ウェーブレット変換図を保存
    plt.savefig(img_file_name)
    plt.show()
    os.chdir('../')
    os.chdir('../')

    # ウェーブレット変換を作成するまでの処理を順番に実行する関数
    # record_list:使用する心電データのファイルパス
    # mode:train用かtest用かの指定
    # normalize:心電データを標準化するかどうか


def preprocess_dataset_core(self, record_list, mode="train", normalize=True):
    for i in range(len(record_list)):
        signal, symbols, positions = self.load_data(record_list[i])
        if normalize:
            signal = self._normalize_signal(signal)
        train_X, train_y, test_X, test_y = self.segment_data(signal, symbols, positions)
        self.ecg_to_WT(train_X, train_y, 'train', record_list[i])
        self.ecg_to_WT(test_X, test_y, 'test', record_list[i])


# MITのデータが入ったフォルダパス
dataset_root = r"\MIT\mit-bih-arrhythmia-database-1.0.0"
# 保存するフォルダパス
save_folder_path = r'D:\color_ecg_wt'
# クラスのインスタンス化
base_ecg = BaseECGDatasetPreprocessor(dataset_root, save_folder_path, window_sec=100, freq_range=0.1, sample_rate=360.)
# クラスの実行
base_ecg.preprocess_dataset(L_record_list, record_list, normalize=True, save_fig=True)
