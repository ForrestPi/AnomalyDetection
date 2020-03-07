from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#学習データ
x_train_s, x_test_s, x_test_b = [], [], []
x_ref, y_ref = [], []

x_train_shape = x_train.shape


for i in range(len(x_train)):
    if y_train[i] == 7:#スニーカーは7
        temp = x_train[i]
        x_train_s.append(temp.reshape((x_train_shape[1:])))
    else:
        temp = x_train[i]
        x_ref.append(temp.reshape((x_train_shape[1:])))
        y_ref.append(y_train[i])

x_ref = np.array(x_ref)

#refデータからランダムに6000個抽出
number = np.random.choice(np.arange(0,x_ref.shape[0]),6000,replace=False)

x, y = [], []

x_ref_shape = x_ref.shape

for i in number:
    temp = x_ref[i]
    x.append(temp.reshape((x_ref_shape[1:])))
    y.append(y_ref[i])

x_train_s = np.array(x_train_s)
x_ref = np.array(x)
y_ref = to_categorical(y)

#テストデータ
for i in range(len(x_test)):
    if y_test[i] == 7:#スニーカーは7
        temp = x_test[i,:,:,:]
        x_test_s.append(temp.reshape((x_train_shape[1:])))

    if y_test[i] == 9:#ブーツは9
        temp = x_test[i,:,:,:]
        x_test_b.append(temp.reshape((x_train_shape[1:])))

x_test_s = np.array(x_test_s)
x_test_b = np.array(x_test_b)



import cv2
from PIL import Image

def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img,dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

X_train_s = resize(x_train_s)
X_ref = resize(x_ref)
X_test_s = resize(x_test_s)
X_test_b = resize(x_test_b)


def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc

#target data
#学習しながら、損失を取得
lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

#reference data
#学習しながら、損失を取得
ld.append(model_r.train_on_batch(batch_ref, batch_y))