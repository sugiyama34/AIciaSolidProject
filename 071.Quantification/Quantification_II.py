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
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
from scipy import optimize

# %matplotlib inline
# -

# # import data

df_data = pd.read_csv(os.path.join('..', 'data', 'qualitization_wanna_buy.csv'))

# +
display(df_data.head())

display(df_data.describe())
# -

# Meanings
#
# - 購買意欲: how the respondant want to buy the item
#     - ◯: want to buy
#     - △: medium
#     - ×: don't want to buy
# - 容積: volume
#     - 1l: 1l
#     - 500ml: 500ml
#     - 300ml: 300ml
# - 形: shape
#     - 円柱: cylinder
#     - 4角柱: quadangular-prism
# - 色: color
#     - 赤: Red
#     - 緑: Green
#     - 青: Blue
# - 回答者: respondant

# # Overview

profile = ProfileReport(df_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

X = pd.get_dummies(df_data[['購買意欲', '容量', '形', '色']])

dependent_vars = ['容量_1l', '容量_500ml', '形_円柱', '色_緑', '色_青']
print(X.columns)
print(dependent_vars)

# # compute variation matrices

# +
# total variation

S_tot = X[dependent_vars].cov(ddof=0)

S_tot

# +
# "within" and "between" variation

N_yes = X['購買意欲_◯'].sum()
N_middle = X['購買意欲_△'].sum()
N_no = X['購買意欲_×'].sum()

S_yes = X[X['購買意欲_◯'] == 1][dependent_vars].cov(ddof=0)
S_middle = X[X['購買意欲_△'] == 1][dependent_vars].cov(ddof=0)
S_no = X[X['購買意欲_×'] == 1][dependent_vars].cov(ddof=0)

S_within = (N_yes * S_yes + N_middle * S_middle + N_no * S_no) / (N_yes + N_middle + N_no)
S_between = S_tot - S_within

S_between
# -

# # Solve maximizing equation

# +
# solve maximizing equation
# Via some equivariant transformation, we find that the maximam eigenvalue is the eta^2 and its eigen vector is the qualitization vector

np.linalg.eig(np.linalg.inv(S_tot).dot(S_between))

# +
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_tot).dot(S_between))

idx_eta_square = np.argmax(eig_vals)
eta_square = eig_vals[idx_eta_square]
qualitization_vector = eig_vecs[:, idx_eta_square]

print('idx_eta_square:', idx_eta_square)
print('eta_square:', eta_square)
print('qualitization_vector:', qualitization_vector)

# -

# # Compute $y$ and Visualize

# +
df_data_with_y = pd.concat([df_data[['購買意欲']], X[dependent_vars]], axis=1)
df_data_with_y['y'] = X[dependent_vars].dot(qualitization_vector)

y_mean_yes = df_data_with_y[df_data_with_y['購買意欲'] == '◯']['y'].mean()
y_mean_no = df_data_with_y[df_data_with_y['購買意欲'] == '×']['y'].mean()
if y_mean_yes < y_mean_no:
    qualitization_vector = -1 * qualitization_vector
    df_data_with_y['y'] = - df_data_with_y['y']
    
df_data_with_y

# +
print('mean of y in each group')
print(df_data_with_y.groupby('購買意欲')['y'].mean())

mapper_label = {'◯': 'yes', '△': 'middle', '×': 'no'}

df_data_with_y.groupby('購買意欲')[['購買意欲', 'y']].apply(lambda x: plt.hist(x['y'], alpha=.3, label=mapper_label[x['購買意欲'].values[0]]))

plt.legend()
plt.show()
# -

# Thus, we have the result of qualitization I as:
#
# $$
# \begin{align}
#     a_\text{1l} &= 0.2665 \\
#     a_\text{500ml} &= 0.7238 \\
#     a_\text{300ml} &= 0 \\
#     a_\text{cylinder} &= 0.2629 \\
#     a_\text{Quadrangular prism} &= 0 \\
#     a_\text{Red} &= 0 \\
#     a_\text{Green} &= -0.1852 \\
#     a_\text{Blue} &= 0.5492 \\
# \end{align}
# $$
#
# We can understand the following facts from the above result:
# - Volume: 500ml is preferred most, 1l is the second
# - Shape: cylinder is better
# - Color: blue is the best, the second is red, and green is the worst
# - The characteristics has capability to explain which water bottles are more likely to be preferred.
# - It is reasonable to say that some types of water bottles won't be best sellers. (items with "×" are placed in left area)
# - It is reasonable to say that characteristics of wanna-buy items may differ (since items with "◯" are dispersed in the graph)
#
# これは、数量化I類の結果として、
#
# $$
# \begin{align}
#     a_\text{1l} &= 0.2665 \\
#     a_\text{500ml} &= 0.7238 \\
#     a_\text{300ml} &= 0 \\
#     a_\text{cylinder} &= 0.2629 \\
#     a_\text{Quadrangular prism} &= 0 \\
#     a_\text{Red} &= 0 \\
#     a_\text{Green} &= -0.1852 \\
#     a_\text{Blue} &= 0.5492 \\
# \end{align}
# $$
#
# という意味になる。
#
#
# これから、次の結果が読み取れる。
# - 容積は、500mlが一番好まれ、次に1lが好まれる。
# - 形は、4角柱より円柱の方が好まれる
# - 色は、青が一番好まれ、次に赤、次に緑である。
# - 水筒の特徴から、購買意欲がかなり説明できる（ヒストグラムより）
# - 買われない商品ははっきりしている（青が左に固まっている）
# - 買われる商品の特徴は、かなりばらついている可能性がある（左の方にも緑がある）
#     - 好みによって、購買意欲が変わる可能性あり？

# # scatter visualization of data

# +
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_tot).dot(S_between))

argsort_ev = np.argsort(eig_vals)

idx_eta_square_2 = argsort_ev[-2]
eta_square_2 = eig_vals[idx_eta_square_2]
qualitization_vector_2 = eig_vecs[:, idx_eta_square_2]

print('idx_eta_square_2:', idx_eta_square_2)
print('eta_square_2:', eta_square_2)
print('qualitization_vector_2:', qualitization_vector_2)


# +
df_data_with_y['y2'] = X[dependent_vars].dot(qualitization_vector_2)
 
df_data_with_y

# +
markers = {
    '◯': 'o',
    '△': '^',
    '×': 'x',
}

for wanna_buy in ['◯', '△', '×']:
    data = df_data_with_y[df_data_with_y['購買意欲'] == wanna_buy]
    plt.scatter(data['y'], data['y2'], s=1000, marker=markers[wanna_buy], alpha=0.1)


# -

# ## scatter visualization of category

def centerize_qualitization(qual):
    y_volume = np.concatenate([qual[:2], [0]])
    y_shape = np.concatenate([qual[2: 3], [0]])
    y_color = np.concatenate([[0], qual[3:]])
    
    y_volume = y_volume - y_volume.mean()
    y_shape = y_shape - y_shape.mean()
    y_color = y_color - y_color.mean()
    
    return np.concatenate([y_volume, y_shape, y_color])


# +
y1 = centerize_qualitization(qualitization_vector)
y2 = centerize_qualitization(qualitization_vector_2)

df_qualitization = pd.DataFrame(
    {
        'category': ['1l', '500ml', '300ml', 'cylinder', 'quad prism', 'red', 'green', 'blue'],
        'y1': y1,
        'y2': y2,
    })

df_qualitization

# +
plt.scatter(df_qualitization['y1'], df_qualitization['y2'], s=100)
texts = [plt.text(df_qualitization['y1'][i], df_qualitization['y2'][i], df_qualitization['category'][i], fontsize=18, ha='center', va='center') for i in range(len(df_qualitization))]
adjust_text(texts)

print("visualiztion of categories")
# -


