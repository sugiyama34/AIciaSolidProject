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
from sklearn.decomposition import FactorAnalysis

# # import data - データの読み込み

# +
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

display(df_scores.head())

display(df_scores.describe())
# -

# # See profile

profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Factor Analysis

# +
n_factors=3

fa = FactorAnalysis(n_components=n_factors, random_state=57)
fa.fit(df_scores)

# +
df_factor_loading = pd.DataFrame(fa.components_.T, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_scores.columns)

df_factor_loading
# -

# # compute Factor score

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
factor_scores = fa.transform(df_sample)  # compute factor scores

pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)
# -


