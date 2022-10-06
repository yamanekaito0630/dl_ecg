import tensorflow as tf
from utils import DatasetCreator

x_train, x_test, y_train, y_test = DatasetCreator("L", "N", "R", "V").create_dataset()

# 同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
new_model = tf.keras.models.load_model('./model')

# 正解率を検査
loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
print("Restored model, loss: {:5.2f}%".format(100 * loss))
