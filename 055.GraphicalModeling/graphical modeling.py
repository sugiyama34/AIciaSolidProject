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
import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
from scipy.stats import chi2

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

# Model selection process:
# 1. Find the smallest (absolute) value in the standardized inverse of correlation matrix
# 2. Assume the pair of variables above are conditionally indepencent
# 3. Estimate correpation matrix with the assumption and see deviance and p-value.
# 4. If p >> 0, repeat the procedure
# 5. If p is small, then stop.
#
# モデル選択は以下の順序で行う：
# 1. 相関行列の逆行列の標準化したものを計算し、（絶対）値最小のものを探す。
# 2. その2つの変数は、条件付き独立であると仮定する
# 3. その仮定のもと、相関行列を推定し、逸脱度とp-値を見る
# 4. p-値が大きければ、上記の手順を続行
# 5. p-値が小さければ終了。

df_videos_restricted = df_videos[['動画時間_s', '視聴回数', 'コメント', '高評価件数', '低評価件数']]
sample_cor = df_videos_restricted.corr()
column_names = df_videos_restricted.columns
estimated_cors = [sample_cor]

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# The correlation between `動画時間_s` and `視聴回数` attains the minimum, thus we assume those variables are conditionally independent.
# Compute `estimated_correlation`, `deviance`, and `p`.
#
# `動画時間_s` と `視聴回数` の相関が、絶対値最小なので、これらが条件付き独立だと仮定。
# 相関行列の推定値と、逸脱度と、p-値を計算する。

# +

cond_ind_pairs = [(0, 1)]

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

# The value of p is enough large -> continue
#
# p-値が充分大きいので、続行。

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(0, 1), (0, 3)]  # the correlation between `動画時間_s`　and `高評価件数` is the smallest

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
cond_ind_pairs = [(0, 1), (0, 3), (0, 4)]  # the correlation between `動画時間_s`　and `低評価件数` is the smallest

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
cond_ind_pairs = [(0, 1), (0, 3), (0, 4), (2, 3)]

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
cond_ind_pairs = [(0, 1), (0, 3), (0, 4), (2, 3), (0, 2)]

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

# It seems that it is not better idea to impose the conditional independence between 0 and 2 since with the condition, 0-th variable is independent with all the other variables.
#
# 0 と 2 の条件付き独立を仮定すると、 0 が他のすべての変数と独立になるので、良くないらしい。

# +
cond_ind_pairs = [(0, 1), (0, 3), (0, 4), (2, 3)]  # remove (0, 2)

estimated_cors.append(estimate(estimated_cors[-2], cond_ind_pairs))  # -1 -> -2

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
cond_ind_pairs = [(0, 1), (0, 3), (0, 4), (2, 3), (2, 4)]

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

# Not so bad.
#
# こちらは悪くない。

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(0, 1), (0, 3), (0, 4), (2, 3), (2, 4), (1, 2)]

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

# The p-value is too small.
# Thus, the list of conditional independence is `[(0, 1), (0, 3), (0, 4), (2, 3), (2, 4)]`
#
# p-値が小さすぎる。
# なので、条件付き独立な変数のリストは `[(0, 1), (0, 3), (0, 4), (2, 3), (2, 4)]` だろう。

# +
# sample correlation
print('sample correlation', '標本相関係数')
display(pd.DataFrame(estimated_cors[0], index=column_names, columns=column_names))

# estimated correlation
print('estimated correlation', '相関係数の推定値')
display(pd.DataFrame(estimated_cors[-2], index=column_names, columns=column_names))

# estimated vs sample
print('estimated vs sample', '値の差')
display(pd.DataFrame(estimated_cors[-2] - estimated_cors[0], index=column_names, columns=column_names))

# compute partial correlation
print('partial correlation', '偏相関係数')
display(pd.DataFrame(inverse_cor(estimated_cors[-2]), index=column_names, columns=column_names).applymap(lambda x: -x if x < 1-1e-4 else 1))
# -

# Thus, we have the following graphical model.
#
# このようにして、次のグラフィカルモデルを手に入れた。
#
# <img src='figure/GraphicalModel.png'>


