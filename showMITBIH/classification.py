from utils import DatasetCreator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = DatasetCreator("L", "N", "R", "V").create_dataset()
width, height = x_train.shape[1:3]
print(width)
print(height)

n_classes = 4
base_model = VGG19(include_top=False, input_shape=(width, height, 3))

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
prediction = layers.Dense(4, activation='softmax')(x)
for layer in base_model.layers[:17]:
    layer.trainable = False
model = models.Model(inputs=base_model.input, outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    validation_split=0.3,
                    batch_size=128, epochs=100, verbose=0)

model.save('./model')

# 正解率の表示
acc = history.history['acc']
val_acc = history.history['val_acc']
nb_epoch = len(acc)
plt.plot(range(nb_epoch), acc, marker='.', label='training')
plt.plot(range(nb_epoch), val_acc, marker='.', label='validation')
plt.title('accuracy of model')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlim(0, 100)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.ylim(0., 1.)
plt.savefig('./evaluated/accuracy.png')
plt.show()

# 損失関数の表示
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(nb_epoch), loss, marker='.', label='training')
plt.plot(range(nb_epoch), val_loss, marker='.', label='validation')
plt.title('loss of model')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.xlim(0, 100)
plt.ylabel('loss')
plt.savefig('./evaluated/loss.png')
plt.show()
