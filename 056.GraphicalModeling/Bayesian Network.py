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

# # This notebook is not prepared for you readers
#
# This is just a notebook for making materials of the movie.
#
# So, it is hard to read and not understandable.
#
# # この notebook は読みやすくなるように書かれていません。
#
# この notebook は動画素材作成用です。
#
# 読みづらいですし、読んでわかるようになっていません。

# +
import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
from scipy.stats import chi2, t

pd.options.display.float_format = '{:.4f}'.format

# +
df_videos = pd.read_csv(os.path.join('..', 'data', 'AIcia_videos.csv'))
df_videos['公開日時'] = pd.to_datetime(df_videos['公開日時'])
df_videos['動画時間_s'] = pd.to_timedelta(df_videos['動画時間']).apply(lambda x: x.seconds)

df_videos = df_videos.drop(['動画時間'], axis=1)

df_videos.head()
# -

profiling = ProfileReport(df_videos)

profiling


# # Implement Graphical Modeling

def inverse_cor(cor):
    cor_inv = np.linalg.inv(cor)
    diag = np.diag(1/np.sqrt(np.diag(cor_inv)))
    return diag.dot(cor_inv).dot(diag)


def estimate(sample_cor, cond_ind_pairs, error_torelance=1e-4, verbose=1):
    '''Estimate correlation matrix from sample correlation and pairs of indices with conditional independence

    Parameters
    ----------
    sample_cor : 2d-array
        sample correlation matric
    arg2 : list of pairs of int
        list of pairs of indices.
        if (i, j) is in the list, this means that the i-th and j-th variables are conditionally independent given all the other variables.
    error_torelance : float
        torelance of error
    verbose : int
        verbose
    
    Returns
    -------
    estimated_cor : 2d-array
        estimated correlation matrix
    
    '''
    dim = sample_cor.shape[0]
    sample_cor = np.array(sample_cor)
    
    estimated_cor = sample_cor.copy()    
    error = 1
    if verbose > 0:
        counter = 0
    
    while error > error_torelance:
        if verbose > 0:
            print('========', counter, '-th iteration', '========')
            counter = counter + 1
        
        estimated_cor_before = estimated_cor.copy()
        for i, j in cond_ind_pairs:
            estimated_cor_inv = np.linalg.inv(estimated_cor)
            new_ij = estimated_cor[i][j] + estimated_cor_inv[i][j]/(estimated_cor_inv[i][i] * estimated_cor_inv[j][j] - estimated_cor_inv[i][j]**2)
            estimated_cor[i][j] = new_ij
            estimated_cor[j][i] = new_ij
        
        error = np.abs(estimated_cor - estimated_cor_before).max().max()
    
    return estimated_cor


def deviance_and_p(original_cor, estimated_cor, df):
    dim = original_cor.shape[0]
    dev = dim * (np.log(np.linalg.det(estimated_cor)) - np.log(np.linalg.det(original_cor)))
    p = 1 - chi2.cdf(dev, df)
    
    return dev, p


# # Selecting models

# 後で書く

# ## Step1 : ['動画時間_s'] and ['視聴回数']

# +
df_videos_restricted_1 = df_videos[['動画時間_s', '視聴回数']]
cor = df_videos_restricted_1.corr().iloc[0, 1]
n = len(df_videos_restricted_1.index)
t_value = cor * np.sqrt((n-1)/(1-cor**2))

print('cor')
print(cor)

print('p-value')
print(np.min([t.cdf(t_value, n-2), 1-t.cdf(t_value, n-2)]))
# -

# ## Step 2 : all variables

df_videos_restricted = df_videos[['動画時間_s', '視聴回数', 'コメント', '高評価件数', '低評価件数']]
sample_cor = df_videos_restricted.corr()
column_names = df_videos_restricted.columns
estimated_cors = [sample_cor]

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

# p-値が十分大きいので続行

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3), (0, 4)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

# p-値が十分大きいので続行

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3), (0, 4), (2, 4)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

# p-値が十分大きいので続行

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3), (0, 4), (2, 4), (0, 3)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3), (0, 4), (2, 4), (0, 3), (0, 2)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(2, 3), (0, 4), (2, 4), (0, 3), (0, 2), (1, 2)]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=column_names, columns=column_names)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', p)
# -

# p-値が小さいので終了


