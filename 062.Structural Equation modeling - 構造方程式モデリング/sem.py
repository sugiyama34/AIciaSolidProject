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

# !pip install semopy

import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
import semopy

# # import data - データの読み込み

# +
df_scores = pd.read_csv(os.path.join('..', 'data', '12_subject_scores.csv'))

display(df_scores.head())

display(df_scores.describe())
# -

# # See profile

profile = ProfileReport(df_scores, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Structural Equation Modeling

# +
# Define 
mod = """\
lang =~ 国語 + 英語 + 倫理
math =~ 数学 + 物理 + 化学
memory =~ 生物 + 地学 + 日本史 + 世界史 + 地理 + 経済
math ~ lang
memory ~ lang
math ~~ memory
"""

model = semopy.Model(mod)
# -

# load data
model.load_dataset(df_scores)

opt = semopy.Optimizer(model)
objective_function_value = opt.optimize()
objective_function_value

# +
df_result = semopy.inspect(opt)

df_result
# -

df_trivial_result = pd.DataFrame(
    {
        'lval': ['lang', 'math', 'memory',],
        'op': ['=~', '=~', '=~',],
        'rval': ['国語', '数学', '生物',],
        'Value': [1, 1, 1,],
        'SE': [0, 0, 0,],
        'Z-score': [np.inf, np.inf, np.inf,],
        'P-value': [0, 0, 0,],
    }
)
df_trivial_result

# +
df_result = pd.concat([df_result, df_trivial_result], axis=0, ignore_index=True)

df_result
# -

# ## Compute standardized scores

# +
df_result_structural = df_result[df_result['op'] == '~']
df_result_measurement = df_result[df_result['op'] == '=~']
df_result_variance = df_result[df_result['op'] == '~~']
dropping_index = df_result_variance[(df_result_variance['lval'] == 'math') & (df_result_variance['rval'] == 'memory')].index.values[0]
df_result_math_memory = df_result.iloc[dropping_index, :]
df_result_variance = df_result_variance.drop(dropping_index, axis=0)

print('df_result_structural')
display(df_result_structural)

print('df_result_measurement')
display(df_result_measurement)

print('df_result_math_memory')
display(df_result_math_memory)

print('df_result_variance')
display(df_result_variance)

# +
# step 1 variance of lang
df_result_variance['variable'] = df_result_variance['lval']
df_result_variance = df_result_variance.set_index('variable')
df_result_variance['variance'] = np.nan

lang_var = df_result_variance.loc['lang', 'Value']
df_result_variance.loc['lang', 'variance'] = lang_var

df_result_variance


# +
# step 2 variance of math, memory

def compute_var_of_math_memory(target):
    coef = df_result_structural[df_result_structural['lval'] == target]['Value'].values[0]
    var = df_result_variance[df_result_variance['lval'] == target]['Value'].values[0] + lang_var * coef**2
    return var

var_math = compute_var_of_math_memory('math')
var_memory = compute_var_of_math_memory('memory')

df_result_variance.loc['math', 'variance'] = var_math
df_result_variance.loc['memory', 'variance'] = var_memory

df_result_variance

# +
# step 3 variance of manifest variables

map_latent_manifest = {
    '国語': 'lang',
    '英語': 'lang',
    '倫理': 'lang',
    '数学': 'math',
    '物理': 'math',
    '化学': 'math',
    '生物': 'memory',
    '地学': 'memory',
    '日本史': 'memory',
    '世界史': 'memory',
    '地理': 'memory',
    '経済': 'memory',
}

def compute_var_of_manifest_var(target):
    latent = map_latent_manifest[target]
    latent_var = df_result_variance.loc[latent, 'variance']
    coef = df_result_measurement[(df_result_measurement['rval'] == target)]['Value'].values[0]
    var = df_result_variance[df_result_variance['lval'] == target]['Value'].values[0] + latent_var * coef**2
    return var

for key in map_latent_manifest.keys():
    df_result_variance.loc[key, 'variance'] = compute_var_of_manifest_var(key)

df_result_variance

# +
# step 4 std coef of structural part

df_result_structural = df_result_structural.merge(df_result_variance[['variance']], how='left', left_on='lval', right_index=True).rename(columns={'variance': 'lvariance'})
df_result_structural = df_result_structural.merge(df_result_variance[['variance']], how='left', left_on='rval', right_index=True).rename(columns={'variance': 'rvariance'})
df_result_structural['std_coef'] = df_result_structural['Value'] / np.sqrt(df_result_structural['lvariance']) * np.sqrt(df_result_structural['rvariance'])

display(df_result_structural)

df_result_math_memory = pd.DataFrame(df_result_math_memory).T
df_result_math_memory = df_result_math_memory.merge(df_result_variance[['variance']], how='left', left_on='lval', right_index=True).rename(columns={'variance': 'lvariance'})
df_result_math_memory = df_result_math_memory.merge(df_result_variance[['variance']], how='left', left_on='rval', right_index=True).rename(columns={'variance': 'rvariance'})
df_result_math_memory['std_coef'] = df_result_math_memory['Value'] / np.sqrt(df_result_math_memory['lvariance']) / np.sqrt(df_result_math_memory['rvariance'])

display(df_result_math_memory)
# -

df_result_measurement

# +
# step 5 std coef of measurement part

df_result_measurement = df_result_measurement.merge(df_result_variance[['variance']], how='left', left_on='lval', right_index=True).rename(columns={'variance': 'lvariance'})
df_result_measurement = df_result_measurement.merge(df_result_variance[['variance']], how='left', left_on='rval', right_index=True).rename(columns={'variance': 'rvariance'})
df_result_measurement['std_coef'] = df_result_measurement['Value'] * np.sqrt(df_result_measurement['lvariance']) / np.sqrt(df_result_measurement['rvariance'])

df_result_measurement

# +
# step 6 std coef of manifest variable part

df_result_variance = df_result_variance.rename(columns={'variance': 'lvariance'})
df_result_variance['rvariance'] = df_result_variance['lvariance']

df_result_variance['std_coef'] = np.sqrt(df_result_variance['Value'] / df_result_variance['lvariance'])

df_result_variance

# +
# step 7 unionize

df_result_std = pd.concat([df_result_structural, df_result_math_memory, df_result_measurement, df_result_variance])

df_result_std
# -


