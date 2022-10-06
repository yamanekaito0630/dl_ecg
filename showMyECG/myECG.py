import numpy as np
import utils as u

# 使用するテキストデータを指定(BiTalino)
signal = np.loadtxt('beat/squat2.txt', comments='#')[:, 5]
move_name = 'スクワット'

# 抽出時間
start = 10
end = 20

u.show_my_ecg(signal, move_name, start, end)
u.show_my_furie(signal, move_name, start, end)
u.show_my_wavelet(signal, move_name, start, end)
