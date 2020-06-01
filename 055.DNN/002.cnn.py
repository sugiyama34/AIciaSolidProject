# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# # load data

from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# +
num_row = 2
num_col = 5

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(10):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title('Label: {}'.format(y_train[i]))
plt.tight_layout()
plt.show()
# -

# # build model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape

# +
x = Input(shape=(28, 28), name='Input')  # shape of input
z = Reshape((28, 28, 1), name='Reshape')(x)  # 28x28 -> 28x28x1
z = Conv2D(32, 3, name='Conv_0')(z)  # 28x28x1 -> 26x26x32
z = MaxPooling2D((2, 2), strides=(1, 1), name='Pool_0')(z)  # 26x26x32 -> 13x13x32
z = Conv2D(64, 3, name='Conv_1')(z)  # 13x13x32 -> 11x11x64
z = MaxPooling2D((3, 3), strides=(2, 2), name='Pool_1')(z)  # 11x11x64 -> 5x5x64
z = Flatten()(z)  # 5x5x64 -> 1600
z = Dense(units=128, activation='relu')(z)  # dense + ReLU
p = Dense(units=10, activation='softmax')(z)  # dense + softmax

model = Model(
    inputs=x,
    outputs=p,
)  # build DNN model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])  # compile model
# -

# callbacks
callbacks = [
    EarlyStopping(patience=3),
    ModelCheckpoint(filepath=os.path.join('models', 'CNN', 'test.h5'), save_best_only=True),
]

# train
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)

# +
# see accuracy

accuracy_score(y_test, model.predict(x_test).argmax(axis=-1))
# -

# このモデルの精度は 97.49% です。
# Dense layer のみのものより高精度！

# # compare models

# +
from collections import OrderedDict

class CNNModel:
    def __init__(self, hid_dim_0=32, hid_dim_1=64):
        self.input = Input(shape=(28, 28), name='Input')  # shape of input
        self.reshape = Reshape((28, 28, 1), name='Reshape')  # 28x28 -> 28x28x1
        self.layers = OrderedDict()
        self.layers['conv_0'] = Conv2D(hid_dim_0, 3, name='Conv_0')  # 28x28x1 -> 26x26xhid_dim_0
        self.layers['pool_0'] = MaxPooling2D((2, 2), strides=(1, 1), name='Pool_0')  # 26x26xhid_dim_0 -> 13x13xhid_dim_0
        self.layers['conv_1'] = Conv2D(hid_dim_1, 3, name='Conv_1')  # 13x13xhid_dim_0 -> 11x11xhid_dim_1
        self.layers['pool_1'] = MaxPooling2D((3, 3), strides=(2, 2), name='Pool_1')  # 11x11xhid_dim_1 -> 5x5xhid_dim_1
        self.layers['flatten'] = Flatten()
        self.layers['dense_0'] = Dense(units=128, activation='relu')  # dense + ReLU
        self.last = Dense(units=10, activation='softmax', name='last')
    
    
    def build(self):
        x = self.input
        z = self.reshape(x)
        for layer in self.layers.values():
            z = layer(z)
        p = self.last(z)
        
        model = Model(inputs=x, outputs=p)
        
        return model
# -

dim_hidden_layres = [2**i for i in range(4, 8)]

# +
df_accuracy = pd.DataFrame()

for hid_dim_0 in dim_hidden_layres:
    for hid_dim_1 in dim_hidden_layres:
        print('========', 'hid_dim_0:', hid_dim_0, '; hid_dim_1:', hid_dim_1, '========')
        model = CNNModel(hid_dim_0=hid_dim_0, hid_dim_1=hid_dim_1)
        model = model.build()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath=os.path.join('models', 'CNN', 'model_{}_{}.h5'.format(hid_dim_0, hid_dim_1)), save_best_only=True),
        ]
        n_param = model.count_params()
        model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)
        acc = accuracy_score(y_test, model.predict(x_test).argmax(axis=-1))
        
        df_accuracy = pd.concat([df_accuracy, pd.DataFrame([[hid_dim_0, hid_dim_1, n_param, acc]], columns=['hid_dim_0', 'hid_dim_1', 'n_param', 'accuracy'])])
# -

display(df_accuracy.set_index(['hid_dim_0', 'hid_dim_1'])[['n_param']].unstack())
display(df_accuracy.set_index(['hid_dim_0', 'hid_dim_1'])[['accuracy']].unstack())

df_accuracy.to_csv('cnn_results.csv')


