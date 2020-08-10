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
df_data = pd.read_csv(os.path.join('..', 'data', 'qualitization_wanna_buy.csv'))

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

profile = ProfileReport(df_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # apply qualitization_III

df_data['item_type'] = df_data['容量'] + '_' + df_data['形'] + '_' + df_data['色']
df_data['response'] = (df_data['購買意欲'] == '◯').astype(int)
df_response = df_data.set_index(['回答者', 'item_type'])[['response']].unstack()
df_response.head()

# +
# remove respondent, item with no positive response
df_response = df_response.loc[df_response.sum(axis=1) > 0, :]
df_response = df_response.loc[:, df_response.sum(axis=0) > 0]

df_response.head()

# +
nda_respondent = np.diag(df_response.sum(axis=1))
nda_response = df_response.values
nda_item = np.diag(df_response.sum(axis=0))

nda_respondent_half_inv = np.diag(df_response.sum(axis=1)**(-1/2))
nda_item_half_inv = np.diag(df_response.sum(axis=0)**(-1/2))

nda_standardized_response = nda_respondent_half_inv.dot(nda_response).dot(nda_item_half_inv)

print('========', 'nda_respondent', '========')
display(nda_respondent)
print('========', 'nda_response', '========')
display(nda_response)
print('========', 'nda_item', '========')
display(nda_item)
# print('========', 'nda_respondent_half_inv', '========')
# display(nda_respondent_half_inv)
# print('========', 'nda_item_half_inv', '========')
# display(nda_item_half_inv)
# print('========', 'nda_standardized_response', '========')
# display(nda_standardized_response)

# +
# apply SVD
u, s, vh = np.linalg.svd(nda_standardized_response)

# print('========', 'u', '========')
# display(u)
# print('========', 's', '========')
# display(s)
# print('========', 'vh', '========')
# display(vh)

# +
# qualitization vector and eigen values

qualitization_vector_respondent = nda_respondent_half_inv.dot(u)
qualitization_vector_item = nda_item_half_inv.dot(vh.T)

eigen_values = s**2

# print('========', 'qualitization_vector_respondent', '========')
# display(qualitization_vector_respondent)
# print('========', 'qualitization_vector_item', '========')
# display(qualitization_vector_item)
print('========', 'eigen_values', '========')
display(eigen_values)

# +
df_qual_vec_respondent = pd.DataFrame(qualitization_vector_respondent, index=df_response.index)
df_qual_vec_item = pd.DataFrame(qualitization_vector_item, index=[b for a, b in df_response.columns])

print('========', 'df_qual_vec_respondent', '========')
display(df_qual_vec_respondent)
print('========', 'df_qual_vec_item', '========')
display(df_qual_vec_item)
# -

# # visualize

# ## scatter plot of respondents vs items

# +
df_data_with_1st_qual = df_data.merge(df_qual_vec_respondent[[1]], how='left', left_on='回答者', right_index=True)
df_data_with_1st_qual = df_data_with_1st_qual.merge(df_qual_vec_item[[1]], how='left', left_on='item_type', right_index=True)
df_data_with_1st_qual = df_data_with_1st_qual.rename(columns={'1_x': 'respondent_1st_qual', '1_y': 'item_1st_qual'})

plt.scatter(df_data_with_1st_qual['respondent_1st_qual'], df_data_with_1st_qual['item_1st_qual'], alpha=0.1)

plt.xlabel('Respondent')
plt.ylabel('Item')

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
# -

# ## scatter plot of respondents and items

# +
plt.figure(figsize=(20,15))
plt.scatter(df_qual_vec_respondent[1], df_qual_vec_respondent[2], s=100)
texts = [plt.text(df_qual_vec_respondent[1][i], df_qual_vec_respondent[2][i], df_qual_vec_respondent.index[i], fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_respondent))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

print("visualiztion of respondents")

# +
map_shape = {'円柱': 'cyl', '4角柱': 'quad'}
map_color = {'赤': 'R', '緑': 'G', '青': 'B'}

def label_ja2en(s):
    l = s.split('_')
    return '_'.join([l[0]] + [map_shape[l[1]]] + [map_color[l[2]]])


# +
plt.figure(figsize=(20,15))
plt.scatter(df_qual_vec_item[1], df_qual_vec_item[2], s=100)
texts = [plt.text(df_qual_vec_item[1][i], df_qual_vec_item[2][i], label_ja2en(df_qual_vec_item.index[i]), fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_item))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

print("visualiztion of items")

# +
plt.figure(figsize=(20,15))
plt.scatter(df_qual_vec_respondent[1], df_qual_vec_respondent[2], s=100, c='r')
texts = [plt.text(df_qual_vec_respondent[1][i], df_qual_vec_respondent[2][i], df_qual_vec_respondent.index[i], fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_respondent))]
adjust_text(texts)


plt.scatter(df_qual_vec_item[1], df_qual_vec_item[2], s=100, c='g')
texts = [plt.text(df_qual_vec_item[1][i], df_qual_vec_item[2][i], label_ja2en(df_qual_vec_item.index[i]), fontsize=18, ha='center', va='center') for i in range(len(df_qual_vec_item))]
adjust_text(texts)

plt.axhline(y=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
plt.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')

print("visualiztion of respondents")
# -

# ## see data more

# ### Color and respondent

df_data.groupby(['回答者', '色'])['response'].sum().unstack()

# We find...
# - everybody like blue
# - only C and E prefer red to blue
# - B and D tends to like green
#
# わかること：
# - みんな青好き
# - C と E だけ赤も好む
# - D と D だけ緑も好む

# ### Shape and resopndent

df_data.groupby(['回答者', '形'])['response'].sum().unstack()

# We find...
# - A-E like quadrangular prism (4角柱)
# - F-L like cylinder (円柱)
#
# わかること：
# - A-E は4角柱好き
# - F-L は円柱好き

# ### volume and respondent

df_data.groupby(['回答者', '容量'])['response'].sum().unstack()

# We find...
# - everybody like 500ml
# - A-E like 1l too
#
# わかること：
# - 全員 500ml は好き
# - A-E は 1l も好き


