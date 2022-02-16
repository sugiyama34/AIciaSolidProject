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

import os
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.cross_decomposition import CCA

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


