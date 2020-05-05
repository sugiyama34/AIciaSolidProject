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
from scipy.stats import chi2

pd.options.display.float_format = '{:.4f}'.format

# +
df = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

df.head()

# +
cor_inv = np.linalg.inv(df.corr())
sqrt_diag_inv = np.diag(1/np.sqrt(np.diag(cor_inv)))

cor_inv_std = sqrt_diag_inv.dot(cor_inv).dot(sqrt_diag_inv)
# -

pd.DataFrame(cor_inv_std, index=df.columns, columns=df.columns)

df_cor = pd.read_csv(os.path.join('..', 'data', 'cor.csv'))
df_cor = df_cor.drop(['Unnamed: 0', '公開日時', 'ln_経過日数', 'ln_視聴回数', 'ln_高評価件数'], axis=1).drop([1, 8, 9, 10])

df_cor

np.linalg.inv(df_cor)

df_cor.shape

# +
cor_inv = np.linalg.inv(df_cor)
sqrt_diag_inv = np.diag(1/np.sqrt(np.diag(cor_inv)))

cor_inv_std = sqrt_diag_inv.dot(cor_inv).dot(sqrt_diag_inv)
# -

pd.DataFrame(cor_inv_std, index=df_cor.columns, columns=df_cor.columns)

pd.DataFrame(cor_inv_std, index=df_cor.columns, columns=df_cor.columns).applymap(lambda x: -x/np.sqrt(1+x**2) if x < 1 else 1)


def inverse_cor(cor):
    cor_inv = np.linalg.inv(cor)
    diag = np.diag(1/np.sqrt(np.diag(cor_inv)))
    return diag.dot(cor_inv).dot(diag)


# +
def estimate(sample_cor, cond_ind_pairs, error_torelance=1e-4):
    dim = sample_cor.shape[0]
    sample_cor = np.array(sample_cor)
    
    estimated_cor = sample_cor.copy()
    
#     print('========', 'dim', '========')
#     print(dim)
#     print('========', 'estimated_cor', '========')
#     print(estimated_cor)
    
    error = 1
    
#     print('========', 'error', '========')
#     print(error)
    
    counter = 0
    
    while error > error_torelance:
        print('========', counter, '-th iteration', '========')
        counter = counter + 1
        
        estimated_cor_before = estimated_cor.copy()
        for i, j in cond_ind_pairs:
            print('    ', '========', i, j, '========')
            estimated_cor_inv = np.linalg.inv(estimated_cor)
            new_ij = estimated_cor[i][j] + estimated_cor_inv[i][j]/(estimated_cor_inv[i][i] * estimated_cor_inv[j][j] - estimated_cor_inv[i][j]**2)
            estimated_cor[i][j] = new_ij
            estimated_cor[j][i] = new_ij
        
        error = np.abs(estimated_cor - estimated_cor_before).max().max()
    
    return estimated_cor


estimated_cor = estimate(df_cor.values, [(1, 4), ])

print('========', 'original', '========')
df_original = pd.DataFrame(df_cor.values, index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cor, index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)

pd.DataFrame(inverse_cor(estimated_cor), index=df_cor.columns, columns=df_cor.columns)
# -

estimate(df_cor.values, [])

chi2.cdf(0, 1)

chi2.cdf(10, 1)


def deviance_and_p(original_cor, estimated_cor, df):
    dim = original_cor.shape[0]
    dev = dim * (np.log(np.linalg.det(estimated_cor)) - np.log(np.linalg.det(original_cor)))
    p = chi2.cdf(dev, df)
    
    return dev, p


deviance_and_p(df_cor.values, estimated_cor, df=1)

# ## Do Graphical Modeling

print('inverse of correlation')
pd.DataFrame(inverse_cor(df_cor.values), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), ]
estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

pd.DataFrame(inverse_cor(estimated_cor), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1- p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1- p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6), (1, 6)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6), (1, 6), (2, 4)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6), (1, 6), (2, 4), (4, 6)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6), (1, 6), (2, 4), (4, 6), (5, 6)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=df_cor.columns, columns=df_cor.columns)

# +
cond_ind_pairs = [(1, 4), (4, 5), (0, 1), (0, 4), (0, 5), (1, 5), (1, 3), (0, 3), (2, 5), (2, 3), (2, 6), (0, 6), (1, 6), (2, 4), (4, 6), (5, 6), (3, 4)]
# estimated_cors = [df_cor.values]

estimated_cors.append(estimate(estimated_cors[-1], cond_ind_pairs))

print('========', 'original', '========')
df_original = pd.DataFrame(estimated_cors[0], index=df_cor.columns, columns=df_cor.columns)
display(df_original)

print('========', 'estimated', '========')
df_estimated = pd.DataFrame(estimated_cors[-1], index=df_cor.columns, columns=df_cor.columns)
display(df_estimated)


print('========', 'test rel to original', '========')
dev, p = deviance_and_p(estimated_cors[0], estimated_cors[-1], df=len(cond_ind_pairs))
print('dev:', dev)
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

# ## 2nd trial

# +
idxs = [0, 3, 4, 5, 6]
column_names = df_cor.columns[idxs]
estimated_cors = [df_cor.values[idxs][:, idxs]]
cond_ind_pairs = []

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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(0, 1), (0, 3)]

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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

# +
cond_ind_pairs = [(0, 1), (0, 3), (0, 4)]

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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
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
print('p:', 1 - p)

print('========', 'test rel to prev', '========')
dev, p = deviance_and_p(estimated_cors[-2], estimated_cors[-1], df=1)
print('dev:', dev)
print('p:', 1 - p)
# -

print('inverse of correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names)

print('partial correlation')
pd.DataFrame(inverse_cor(estimated_cors[-1]), index=column_names, columns=column_names).applymap(lambda x: -x/(1 + x**2) if x < 1-1e-4 else 1)


