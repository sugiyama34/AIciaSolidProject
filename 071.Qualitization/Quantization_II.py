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

# +
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

# # Overview

profile = ProfileReport(df_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

X = pd.get_dummies(df_data[['購買意欲', '容量', '形', '色']])

dependent_vars = [x for x in X.columns if x[:4] != '購買意欲']
print(X.columns)
print(dependent_vars)

# +
S_tot = X[dependent_vars].cov()

S_tot

# +
N_yes = df_data['購買意欲'].value_counts()['◯']
N_middle = df_data['購買意欲'].value_counts()['△']
N_no = df_data['購買意欲'].value_counts()['×']

S_yes = X[X['購買意欲_◯'] == 1][dependent_vars].cov()
S_middle = X[X['購買意欲_△'] == 1][dependent_vars].cov()
S_no = X[X['購買意欲_×'] == 1][dependent_vars].cov()

S_between = (N_yes * S_yes + N_middle * S_middle + N_no * S_no) / (N_yes + N_middle + N_no)

S_between


# +
def equation(rho):
    return np.linalg.det(S_tot - rho * S_between)

rho = optimize.newton(equation, 1)
# -

qualitization_vector = np.linalg.eig(S_tot - rho * S_between)[1][:, 0]
qualitization_vector

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
#     a_\text{1l} &= -0.0869 \\
#     a_\text{500ml} &= 0.4135 \\
#     a_\text{300ml} &= -0.3266 \\
#     a_\text{円柱} &= 0.3443 \\
#     a_\text{4角柱} &= -0.3443 \\
#     a_\text{緑} &= 0.1981 \\
#     a_\text{赤} &= 0.3586 \\
#     a_\text{青} &= -0.5567 \\
# \end{align}
# $$
#
# We can understand the following facts from the above result:
# - TO BE WRITTRN
#
# これは、数量化I類の結果として、
#
# $$
# \begin{align}
#     a_\text{1l} &= -0.0869 \\
#     a_\text{500ml} &= 0.4135 \\
#     a_\text{300ml} &= -0.3266 \\
#     a_\text{円柱} &= 0.3443 \\
#     a_\text{4角柱} &= -0.3443 \\
#     a_\text{緑} &= 0.1981 \\
#     a_\text{赤} &= 0.3586 \\
#     a_\text{青} &= -0.5567 \\
# \end{align}
# $$
#
# という意味になる。
#
#
# これから、次の結果が読み取れる。
# - あとでかく


