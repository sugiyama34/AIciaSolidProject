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
# Run this cell if necessary

# !pip install factor_analyzer
# -

import os
import pandas as pd
from pandas_profiling import ProfileReport
from factor_analyzer import FactorAnalyzer

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

# ## factor analysis without rotation

# +
n_factors=3

fa_wo_rotation = FactorAnalyzer(rotation=None, n_factors=n_factors)
# -

# ### compute factor loadings

fa_wo_rotation.fit(df_scores)

pd.DataFrame(fa_wo_rotation.loadings_, index=df_scores.columns, columns=['factor_{}'.format(i) for i in range(n_factors)])

# ### compute factor score

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
factor_scores = fa_wo_rotation.transform(df_sample)  # compute factor scores

pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)
# -

# ## factor analysis with varimax rotation

fa_varimax = FactorAnalyzer(rotation='varimax', n_factors=n_factors)

# ### compute factor loadings

fa_varimax.fit(df_scores)

pd.DataFrame(fa_varimax.loadings_, index=df_scores.columns, columns=['factor_{}'.format(i) for i in range(n_factors)])

# ### compute factor score

# +
factor_scores = fa_varimax.transform(df_sample)  # compute factor scores

pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)
# -

# ## factor analysis with promax rotation

fa_promax = FactorAnalyzer(rotation='promax', n_factors=n_factors)

# ### compute factor loadings

fa_promax.fit(df_scores)

pd.DataFrame(fa_promax.loadings_, index=df_scores.columns, columns=['factor_{}'.format(i) for i in range(n_factors)])

# ### compute factor scores

# +
factor_scores = fa_promax.transform(df_sample)  # compute factor scores

pd.DataFrame(factor_scores, columns=['factor_{}'.format(i) for i in range(n_factors)], index=df_sample.index)
# -


