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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# +
df_result_dnn = pd.read_csv('dnn_results.csv')
df_result_dnn['type'] = 'dnn'
df_result_dnn = df_result_dnn[['type', 'n_param', 'accuracy']]

df_result_cnn = pd.read_csv('cnn_results.csv')
df_result_cnn['type'] = 'cnn'
df_result_cnn = df_result_cnn[['type', 'n_param', 'accuracy']]

# +
df_result_dnn['ln_param'] = np.log(df_result_dnn['n_param'] + 1)
df_result_dnn['log_odds'] = np.log(df_result_dnn['accuracy']/(1-df_result_dnn['accuracy']))

df_result_cnn['ln_param'] = np.log(df_result_cnn['n_param'] + 1)
df_result_cnn['log_odds'] = np.log(df_result_cnn['accuracy']/(1-df_result_cnn['accuracy']))
# -

plt.scatter(df_result_dnn['ln_param'], df_result_dnn['log_odds'], label='dnn')
plt.scatter(df_result_cnn['ln_param'], df_result_cnn['log_odds'], label='cnn')
plt.xlabel('ln_param')
plt.ylabel('accuracy (log odds)')
plt.legend()
plt.show()


