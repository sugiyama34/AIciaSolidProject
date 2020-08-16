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
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport

# %matplotlib inline
# -

# # import data 

# +
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

display(df_scores.head())

display(df_scores.describe())
# -

profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

mapper_ja2en = {
    '国語': 'Japanese',
    '英語': 'English',
    '数学': 'Mathmatics',
    '物理': 'Physics',
    '化学': 'Chemistry',
    '生物': 'Biology',
    '地学': 'Earth',
    '世界史': 'World history',
    '日本史': 'Japanese history',
    '経済': 'Economics',
    '地理': 'Geography',
    '倫理': 'Ethics',
}

# # apply qualitization

# +
df_corr = df_scores.corr()

df_corr

# +
# set up the matrix corresponding with our quadratic target function

nda_diag = - np.diag((df_corr.sum(axis=0).values - 1) + (df_corr.sum(axis=1).values - 1))
nda_non_diag_pre = (df_corr + df_corr.T).values
nda_non_diag = nda_non_diag_pre - np.diag(np.diag(nda_non_diag_pre))

nda_matrix = nda_diag + nda_non_diag

# +
eig_val, eig_vec = np.linalg.eig(nda_matrix)

display(eig_val)
display(eig_vec)

# +
argsort = np.argsort(-eig_val)

qualification_vector_1 = eig_vec[:, argsort[1]]
qualification_vector_2 = eig_vec[:, argsort[2]]
# -

df_qualitification = pd.DataFrame({'qv_1': qualification_vector_1, 'qv_2': qualification_vector_2}, index=df_scores.columns)
df_qualitification

# +
plt.figure(figsize=(10,7))
plt.scatter(df_qualitification['qv_1'], df_qualitification['qv_2'], s=100)
texts = [plt.text(df_qualitification['qv_1'][i], df_qualitification['qv_2'][i], mapper_ja2en[df_qualitification.index[i]], fontsize=18, ha='center', va='center') for i in range(len(df_qualitification))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

print("visualiztion of items")
# -


