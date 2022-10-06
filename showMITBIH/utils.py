import numpy as np
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# 時間軸を生成する関数
def unit_conversion(val, f=100):
    return val / f


# 時間軸を生成する関数
def mit_conversion(val, f=360):
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


def freq_logspace(start_f: float, stop_f: float, num: int):
    """
    numで指定した個数のデータをlogspaceで取得する
    取得する値の範囲をstart_f, stop_fで指定
    """
    log_base = 10
    low_mul = np.log10(start_f)
    high_mul = np.log10(stop_f)
    return np.logspace(low_mul, high_mul, num=num, base=log_base)


def change(val):
    return round(val, 1)


def readlines_file(file_name):
    # 行毎のリストを返す
    with open(file_name, 'r') as file:
        return file.readlines()


def save_file(file_name, text):
    with open(file_name, 'w') as file:
        file.write(text)


# ウェーブレット変換図を読み込み，trainとtestのデータセットを作成するクラス
class DatasetCreator(object):
    # 初期化関数
    def __init__(self, file_label1, file_label2, file_label3, file_label4):
        # ファイルパスの定義
        self.file_path1_train = "./img/" + file_label1 + "/train/*.png"
        self.file_path1_test = "./img/" + file_label1 + "/test/*.png"
        self.file_path2_train = "./img/" + file_label2 + "/train/*.png"
        self.file_path2_test = "./img/" + file_label2 + "/test/*.png"
        self.file_path3_train = "./img/" + file_label3 + "/train/*.png"
        self.file_path3_test = "./img/" + file_label3 + "/test/*.png"
        self.file_path4_train = "./img/" + file_label4 + "/train/*.png"
        self.file_path4_test = "./img/" + file_label4 + "/test/*.png"

    # 画像を読み込み配列に格納する関数
    @staticmethod
    def read_ecg(file_path):
        data_list = []
        for img_path in glob.glob(file_path):
            img = load_img(img_path)
            arr_img = img_to_array(img)
            py_img = list(arr_img)
            data_list.append(py_img)
        return np.array(data_list)

    # データセットを作成する関数
    def create_dataset(self):
        data_list1_train = self.read_ecg(self.file_path1_train)
        data_list1_test = self.read_ecg(self.file_path1_test)[:1100]
        data_list2_train = self.read_ecg(self.file_path2_train)
        data_list2_test = self.read_ecg(self.file_path2_test)[:1100]
        data_list3_train = self.read_ecg(self.file_path3_train)
        data_list3_test = self.read_ecg(self.file_path3_test)[:1100]
        data_list4_train = self.read_ecg(self.file_path4_train)
        data_list4_test = self.read_ecg(self.file_path4_test)[:1100]

        # 正規化
        x_train = np.concatenate([data_list1_train, data_list2_train, data_list3_train, data_list4_train]) / 255.
        x_test = np.concatenate([data_list1_test, data_list2_test, data_list3_test, data_list4_test]) / 255.

        y_train = np.concatenate(
            [np.zeros(len(data_list1_train)), np.ones(len(data_list2_train)), np.full(len(data_list3_train), 2),
             np.full(len(data_list4_train), 3)])
        y_test = np.concatenate(
            [np.zeros(len(data_list1_test)), np.ones(len(data_list2_test)), np.full(len(data_list3_test), 2),
             np.full(len(data_list4_test), 3)])
        # ラベルをone-hot化
        y_train = to_categorical(y_train, 4)
        y_test = to_categorical(y_test, 4)

        return x_train, x_test, y_train, y_test
