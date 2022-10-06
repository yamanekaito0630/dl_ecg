# input
import glob
from PIL import Image
import numpy as np
from keras import layers, models
from keras.utils import np_utils
from keras_applications.vgg19 import VGG19
from sklearn import model_selection
import matplotlib.pyplot as plt


# 画像データを学習用データ（訓練およびテスト）に変換するクラス
class DatasetCreator(object):
    # classes : list型(["L", "N", "R", "V"])
    # image_size : int型(画像サイズ指定)
    # max_read : int型(読み込む画像の枚数)
    def __init__(self, classes, image_size, max_read):
        self.classes = classes
        self.image_size = image_size
        self.max_read = max_read

    def create_dataset(self):
        x = []
        y = []
        num_classes = len(self.classes)

        for index, class_label in enumerate(self.classes):
            images_dir = "./img/" + class_label + "/train"
            files = glob.glob(images_dir + "/*.png")
            for i, file in enumerate(files):
                if i >= self.max_read:
                    break
                image = Image.open(file)
                image = image.convert("RGB")
                image = image.resize((self.image_size, self.image_size))
                data = np.asarray(image)
                x.append(data)
                y.append(index)
        x = np.array(x)
        y = np.array(y)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

        # 正規化
        x_train = x_train.astype("float") / 255
        x_test = x_test.astype("float") / 255

        # one-hot vector
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)

        return x_train, x_test, y_train, y_test


class CreateModel(object):

    def __init__(self, hold_layer_num, dataset):
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = dataset
        self.width, self.height = self.x_train.shape[1:3]
        self.hold_layer_num = hold_layer_num

    def set_model(self):
        vgg19 = VGG19(include_top=False, input_shape=(self.width, self.height, 3))
        c = vgg19.output
        c = layers.Flatten()(c)
        c = layers.Dense(128, activation='relu')(c)
        c = layers.Dense(64, activation='relu')(c)
        c = layers.Dense(64, activation='relu')(c)
        prediction = layers.Dense(4, activation='softmax')(c)
        for layer in vgg19.layers[:self.hold_layer_num]:
            layer.trainable = False
        self.model = models.Model(input=vgg19.input, output=prediction)
        self.model.summary()







classes = ["L", "N", "R", "V"]
image_size = 312
max_read = 2400

dc = DatasetCreator(classes, image_size, max_read)
dc.create_dataset()
