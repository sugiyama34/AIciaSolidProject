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
from tensorflow.keras.layers import Dense, Input, Flatten

# +
x = Input(shape=(28, 28))  # shape of input
z = Flatten()(x)  # 28x28 -> 784
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
    ModelCheckpoint(filepath=os.path.join('models', 'DNN', 'test.h5'), save_best_only=True),
]

# train
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)

# +
# see accuracy

accuracy_score(y_test, model.predict(x_test).argmax(axis=-1))
# -

# The accuracy of this model is 95.66% (in the author's environment.) Pretty good!
#
# このモデルの精度は（筆者の環境では） 95.66% です。
# 悪くない！

# # compare models

# +
from collections import OrderedDict

class DenseModel:
    def __init__(self, layers=1, hid_dim=128):
        self.input = Input(shape=(28, 28), name='input')
        self.flatten = Flatten(name='flatten')
        self.denses = OrderedDict()
        for i in range(layers):
            name = 'dense_{}'.format(i)
            self.denses[name] = Dense(units=hid_dim, activation='relu', name=name)
        self.last = Dense(units=10, activation='softmax', name='last')
    
    
    def build(self):
        x = self.input
        z = self.flatten(x)
        for dense in self.denses.values():
            z = dense(z)
        p = self.last(z)
        
        model = Model(inputs=x, outputs=p)
        
        return model

# +
dim_hidden_layres = [2**i for i in range(11)]
n_layers = range(1, 4)

df_accuracy = pd.DataFrame()

for layers in n_layers:
    for hid_dim in dim_hidden_layres:
        print('========', 'layers:', layers, '; hid_dim:', hid_dim, '========')
        model = DenseModel(layers=layers, hid_dim=hid_dim)
        model = model.build()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath=os.path.join('models', 'DNN', 'model_{}_{}.h5'.format(layers, hid_dim)), save_best_only=True),
        ]
        n_param = model.count_params()
        model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)
        acc = accuracy_score(y_test, model.predict(x_test).argmax(axis=-1))
        
        df_accuracy = pd.concat([df_accuracy, pd.DataFrame([[layers, hid_dim, n_param, acc]], columns=['layers', 'hid_dim', 'n_param', 'accuracy'])])
# -

display(df_accuracy.set_index(['layers', 'hid_dim'])[['n_param']].unstack())
display(df_accuracy.set_index(['layers', 'hid_dim'])[['accuracy']].unstack())

df_accuracy.to_csv('dnn_results.csv')


