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
from sklearn.decomposition import PCA


# %matplotlib inline
# -

# # import data - データの読み込み

# +
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

display(df_scores.head())

display(df_scores.describe())
# -

# # See Profile

profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Primary Component Analysis

# +
n_components=3

pca = PCA(n_components=n_components, random_state=57)
pca.fit(df_scores)

# +
df_factor_loading = pd.DataFrame(pca.components_.T, columns=['component_{}'.format(i) for i in range(n_components)], index=df_scores.columns)

df_factor_loading
# -

# # compute component scores

# +
# compose sample data
df_sample = pd.DataFrame(
    [
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
        [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ],
    columns=df_scores.columns,
    index=['A', 'B', 'C', 'D']
)

df_sample

# +
component_scores = pca.transform(df_sample)  # compute factor scores

pd.DataFrame(component_scores, columns=['component_{}'.format(i) for i in range(n_components)], index=df_sample.index)
# -
# # explained variance ratio


# +
pca_all = PCA(n_components=12)

pca_all.fit(df_scores)

for i, ratio in enumerate(np.concatenate((np.array([0]), np.cumsum(pca_all.explained_variance_ratio_)))):
    print('{}'.format(i), 'components:', '{:.2f}'.format(ratio*100), '% are explained')
plt.plot(list(range(13)), np.concatenate((np.array([0]), np.cumsum(pca_all.explained_variance_ratio_))), 'o-')
plt.xlabel('# of components')
plt.ylabel('explained variance ratio')
plt.show()
# -


