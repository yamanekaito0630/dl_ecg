# ライブラリのインポート
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils import to_categorical


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
             np.full(len(data_list3_train), 3)])
        y_test = np.concatenate(
            [np.zeros(len(data_list1_test)), np.ones(len(data_list2_test)), np.full(len(data_list3_test), 2),
             np.full(len(data_list3_test), 3)])
        # ラベルをone-hot化
        y_train = to_categorical(y_train, 4)
        y_test = to_categorical(y_test, 4)

        return x_train, x_test, y_train, y_test


# 学習モデルを定義して訓練するクラス
class ModelCreator(object):
    # 初期化関数
    # hold_layer_num: VGG19のフリーズする層数
    # dataset: 学習に使用するデータセット(タプル型)
    def __init__(self, hold_layer_num, dataset):
        self.model, self.now_time, self.model_dir, self.history_model, self.cm = [None, None, None, None, None]
        self.x_train, self.x_test, self.y_train, self.y_test = dataset
        self.width, self.height = self.x_train.shape[1:3]
        self.hold_layer_num = hold_layer_num

    # VGG19を定義する関数
    def set_model(self):
        vgg19 = VGG19(include_top=False, input_shape=(self.width, self.height, 3))
        c = vgg19.output
        c = Flatten()(c)
        c = Dense(128, activation='relu')(c)
        c = Dense(64, activation='relu')(c)
        c = Dense(64, activation='relu')(c)
        prediction = Dense(4, activation='softmax')(c)
        for layer in vgg19.layers[:self.hold_layer_num]:
            layer.trainable = False
        self.model = Model(input=vgg19.input, output=prediction)
        self.model.summary()

    # VGG19を訓練する関数
    def train(self):
        os.chdir(DEFAULT_SAVE_MODEL_PATH)
        self.now_time = datetime.now().strftime('%y%m%d_%H%M')
        # 学習したVGG19の重みを保存するフォルダパス
        self.model_dir = os.path.join(
            'models',
            self.now_time + "VGG19_" + str(weight_hold_vgg) + "_" + str(LR) + "_" + PARENT_FILE_PATH

        )
        os.makedirs(self.model_dir, exist_ok=True)
        dir_weights = os.path.join(self.model_dir, 'weights')
        os.makedirs(dir_weights, exist_ok=True)
        # ネットワークの保存
        model_json = os.path.join(self.model_dir, 'model.json')
        with open(model_json, 'w') as f:
            json.dump(self.model.to_json(), f)
        # Callbacksの設定
        cp_filepath = os.path.join(dir_weights, 'ep_{epoch:02d}_ls_{loss:.1f}.h5')
        cp = ModelCheckpoint(
            cp_filepath,
            monitor='val_acc',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=5
        )

        csv_filepath = os.path.join(self.model_dir, "VGG19_" + str(weight_hold_vgg) + "_" + str(
            LR) + "_" + PARENT_FILE_PATH + '_loss.csv')
        csv = CSVLogger(csv_filepath, append=True)
        # モデルのコンパイル
        self.model.compile(optimizer=Adam(lr=LR, decay=DECAY), loss='categorical_crossentropy', metrics=['accuracy'])
        # 学習開始
        self.history_model = self.model.fit(self.x_train, self.y_train, batch_size=BATCH, epochs=EPOCH,
                                            validation_split=0.3, callbacks=[cp, csv])

    # 学習後のVGG19を評価する関数
    def evaluate(self):
        model_path = os.path.join(DEFAULT_SAVE_MODEL_PATH, self.model_dir)

        # テスト用画像を用いた正解率
        probs = self.model.predict(self.x_test)
        # テスト用画像を用いた混同行列
        self.cm = confusion_matrix(self.y_test.argmax(axis=1), probs.argmax(axis=1))

        hist = self.model.history
        # 正解率の表示
        acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        nb_epoch = len(acc)
        plt.plot(range(nb_epoch), acc, marker='.', label='training')
        plt.plot(range(nb_epoch), val_acc, marker='.', label='validation')
        plt.title('accuracy of model')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlim(0, EPOCH)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.ylim(0., 1.)
        os.chdir(model_path)
        plt.savefig('accuracy.png')
        plt.show()
        self.acs = sum(val_acc) / len(val_acc)

        # 損失関数の表示
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        plt.plot(range(nb_epoch), loss, marker='.', label='training')
        plt.plot(range(nb_epoch), val_loss, marker='.', label='validation')
        plt.title('loss of model')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epoch')
        plt.xlim(0, EPOCH)
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.show()

    # 検証したパラメータを保存する関数
    @staticmethod
    def save_param():
        # 保存するjsonデータ
        json_data = {
            "param": {
                'LR': LR,
                'EPOCH': EPOCH,
                'DECAY': DECAY,
                'BATCH': BATCH,
                "hold_layer_num": weight_hold_vgg
            },
            "confusion_matrix": mc.cm,
            "accuracy_score": mc.acs,
        }
        # jsonデータを保存
        with open('log_file.json', 'w') as f:
            json.dump(json_data, f, indent=4)

    # 学習の全ての流れを実行する関数
    def run(self):
        self.set_model()
        self.train()
        self.evaluate()
        self.save_param()


# 学習済みの重みを保存するフォルダパス
DEFAULT_SAVE_MODEL_PATH = './result_models'

# 設定したパラメータ
LR = [1e-4, 1e-5][1]
FREQ_RANGE = 15
EPOCH = 100
DECAY = 0.1
BATCH = 150
DATA_SEC = 100
weight_hold_vgg = 17

# raw: 生の心電データ
# correct: 3次関数を用いて記録されていない箇所を補間した心電データ
STATE = 'correct'  # ['raw', 'correct'][parame]

# 学習済みの重みを保存するフォルダ名
PARENT_FILE_PATH = "range_" + str(FREQ_RANGE) + "_data_sec_" + str(DATA_SEC)

classify_label1 = "L"
classify_label2 = "N"
classify_label3 = "R"
classify_label4 = "V"

# データセット作成クラスの定義
dc = DatasetCreator(classify_label1, classify_label2, classify_label3, classify_label4)
# モデルクラスの定義
mc = ModelCreator(weight_hold_vgg, dc.create_dataset())
# モデルクラスの実行
mc.run()
