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

# !pip install pymc3

import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
import pymc3 as pm

# # import data - データの読み込み

# +
df_videos = pd.read_csv(os.path.join('..', 'data', 'AIcia_videos.csv'))
df_videos['公開日時'] = pd.to_datetime(df_videos['公開日時'])
df_videos['動画時間_s'] = pd.to_timedelta(df_videos['動画時間']).apply(lambda x: x.seconds)

df_videos = df_videos.drop(['動画時間'], axis=1)

df_videos.head()
# -

profile = ProfileReport(df_videos, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Hierarchical Bayesian Modeling

n_videos = len(df_videos)

# +
with pm.Model() as model:
    # prior to parameters
    alpha_plus = pm.TruncatedNormal('alpha_plus', mu=0, sd=1e-1, lower=0)
    beta_plus = pm.TruncatedNormal('beta_plus', mu=0, sd=1e-1, lower=0)
    gamma_plus = pm.TruncatedNormal('gamma_plus', mu=0, sd=100, lower=0)
#     alpha_minus = pm.TruncatedNormal('alpha_minus', mu=0, sd=100, lower=1e-4)
#     beta_minus = pm.TruncatedNormal('beta_minus', mu=0, sd=100, lower=1e-4)
#     gamma_minus = pm.TruncatedNormal('gamma_minus', mu=0, sd=100, lower=1e-4)
    
    # prior to fun
    fun = pm.Normal('fun', mu=0, sd=1, shape=n_videos)
    
    # play
    raw_play = df_videos['視聴回数']
    e = 1e-4
    play = pm.Uniform('play', lower=raw_play-e, upper=raw_play+e, shape=n_videos)
    
    # +1 and -1
    lambda_plus_pre = (alpha_plus + beta_plus * fun) * play + gamma_plus
    lambda_plus = pm.TruncatedNormal('lambda_plus', mu=lambda_plus_pre, sd=1e-2, lower=1e-5, shape=n_videos)
    like = pm.Poisson('like', mu=lambda_plus, observed=df_videos['高評価件数'])
    
#     lambda_minus = (alpha_minus + beta_minus * fun) * play + gamma_minus
#     dislike = pm.Poisson('dislike', mu=lambda_minus, observed=df_videos['低評価件数'])
    
    trace = pm.sample(1500, tune=1000, chains=5, random_seed=57)

# +
df_trace = pm.summary(trace)

df_trace
# -

df_trace.index.values

df_trace.loc[['alpha_plus', 'beta_plus', 'gamma_plus'], :]

df_trace['mean'].head(63).values

pm.traceplot(trace)

model_map = pm.find_MAP(model=model)
model_map

model_map['fun']

np.var(model_map['fun'])


