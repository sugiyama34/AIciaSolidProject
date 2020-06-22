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

with pm.Model() as model:
    # prior to parameters
    alpha_plus = pm.Normal('alpha_plus', mu=-3, sd=2)
    beta_plus = pm.TruncatedNormal('beta_plus', mu=0, sd=1, lower=0)
    alpha_minus = pm.Normal('alpha_minus', mu=-3, sd=2)
    beta_minus = pm.TruncatedNormal('beta_minus', mu=0, sd=1, upper=0)
    
    # prior to fun
    fun = pm.Normal('fun', mu=0, sd=1, shape=n_videos)
    
    # play
    play = df_videos['視聴回数']
    
    # +1 and -1
    lambda_plus = pm.math.exp((alpha_plus + beta_plus * fun)) * play
    like = pm.Poisson('like', mu=lambda_plus, observed=df_videos['高評価件数'])
    
    lambda_minus = pm.math.exp((alpha_minus + beta_minus * fun)) * play
    dislike = pm.Poisson('dislike', mu=lambda_minus, observed=df_videos['低評価件数'])
    
    trace = pm.sample(1500, tune=1000, chains=5, random_seed=57)

pm.traceplot(trace)

# +
df_trace = pm.summary(trace)

df_trace
# -

model_map = pm.find_MAP(model=model)
model_map

model_map['fun']

np.std(model_map['fun'])

df_trace.loc['fun[0]':'beta_plus', ['mean']].sort_values('mean', ascending=False)

df_videos.iloc[[0, 2, 18, 3, 4]]

df_videos.iloc[[40, 62, 25, 22, 56]]

df_trace.loc['fun[0]':'beta_plus', 'mean'].describe()

df_video

# # fun vs comment

with pm.Model() as model_with_comment:
    # prior to parameters
    alpha_plus = pm.Normal('alpha_plus', mu=-3, sd=2)
    beta_plus = pm.TruncatedNormal('beta_plus', mu=0, sd=1, lower=0)
    alpha_minus = pm.Normal('alpha_minus', mu=-3, sd=2)
    beta_minus = pm.TruncatedNormal('beta_minus', mu=0, sd=1, upper=0)
    alpha_comment = pm.Normal('alpha_comment', mu=-3, sd=2)
    beta_comment = pm.TruncatedNormal('beta_comment', mu=0, sd=1, lower=0)
    
    # prior to fun
    fun = pm.Normal('fun', mu=0, sd=1, shape=n_videos)
    
    # prior to comment
    latent_comment = pm.Normal('latent_comment', mu=0, sd=1, shape=n_videos)
    
    # play
    play = df_videos['視聴回数']
    
    # +1, -1, comment
    lambda_plus = pm.math.exp((alpha_plus + beta_plus * fun)) * play
    like = pm.Poisson('like', mu=lambda_plus, observed=df_videos['高評価件数'])
    
    lambda_minus = pm.math.exp((alpha_minus + beta_minus * fun)) * play
    dislike = pm.Poisson('dislike', mu=lambda_minus, observed=df_videos['低評価件数'])
    
    lambda_comment = pm.math.exp((alpha_comment + beta_comment * latent_comment)) * play
    comment = pm.Poisson('comment', mu=lambda_comment, observed=df_videos['コメント'])
    
    trace = pm.sample(1500, tune=1000, chains=5, random_seed=57)

pm.traceplot(trace)

# +
df_trace = pm.summary(trace)

df_trace
# -

df_latent = df_trace.loc['fun[0]':'latent_comment[62]', ['mean']].reset_index()
df_latent['variable'] = df_latent['index'].apply(lambda x: x.split('[')[0])
df_latent['index'] = df_latent['index'].apply(lambda x: x.split('[')[1].split(']')[0])
df_latent = df_latent.set_index(['index', 'variable']).unstack()

df_latent.describe()

df_latent.corr()


