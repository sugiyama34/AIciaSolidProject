# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

np.random.seed(57)

# %matplotlib inline
# -

# # データの読み込み

# +
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

display(df_scores.head())

display(df_scores.describe())
# -

# # See profile

profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Canonical Correlation Analysis

# +
n_components = 3

cca = CCA(n_components=n_components)

lst_col_science = ['数学', '物理', '化学', '生物', '地学']
lst_col_literature = ['国語', '英語', '世界史', '日本史', '経済', '地理', '倫理']

cca.fit(df_scores[lst_col_science], df_scores[lst_col_literature])
# -

display(pd.DataFrame(cca.x_weights_, index=lst_col_science, columns=['comp_{}'.format(i) for i in range(n_components)]))
display(pd.DataFrame(cca.y_weights_, index=lst_col_literature, columns=['comp_{}'.format(i) for i in range(n_components)]))

cca.x_weights_

# # Common component extraction

# +
n_bin = 1001
coef_signal = 0.01
coef_noise = 1
coef_noise_2 = 1e-4
p_noise_rate = 0.7
q_noise_rate = -0.4

x = np.linspace(0, 10, n_bin)

y = coef_signal * ((np.sin(x) > 1/np.sqrt(2)).astype(int) - (np.sin(x) < -1/np.sqrt(2)).astype(int))

plt.plot(x, y)

# +


p_noise = np.random.randn(n_bin)
p1 = y + coef_noise * p_noise + coef_noise_2 * np.random.randn(n_bin)
p2 = y + p_noise_rate * coef_noise * p_noise + coef_noise_2 * np.random.randn(n_bin)
p = np.concatenate([p1.reshape(-1, 1), p2.reshape(-1, 1)], axis=1)

q_noise = np.random.randn(n_bin)
q1 = y + coef_noise * q_noise + coef_noise_2 * np.random.randn(n_bin)
q2 = y + q_noise_rate * coef_noise * q_noise + coef_noise_2 * np.random.randn(n_bin)
q = np.concatenate([q1.reshape(-1, 1), q2.reshape(-1, 1)], axis=1)

plt.plot(x, p1)
plt.show()
plt.plot(x, p2)
plt.show()
plt.plot(x, q1)
plt.show()
plt.plot(x, q2)
plt.show()

# +
cca = CCA(n_components=1)

cca.fit(p, q)
# -

plt.plot(x, cca.transform(p))

# +
lm = LinearRegression()

lm.fit(p, q)
# -

plt.plot(x, lm.predict(p))

# +
pca_1 = PCA(n_components=1)
pca_2 = PCA(n_components=1)

pca_1.fit(p)
pca_2.fit(np.concatenate([p, q], axis=1))
# -

plt.plot(x, pca_1.transform(p))
plt.show()
plt.plot(x, pca_2.transform(np.concatenate([p, q], axis=1)))


