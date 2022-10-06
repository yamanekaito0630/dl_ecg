import numpy as np


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
