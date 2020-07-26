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

import os
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression

# # import data

df_data = pd.read_csv(os.path.join('..', 'data', 'qualitization_height.csv'))

# +
display(df_data.head())

display(df_data.describe())
# -

# The data is artificially generated from some normal distributions.
# The parameter of the normal distributions are given by [学校保健統計調査 令和元年度 全国表 | ファイル | 統計データを探す | 政府統計の総合窓口](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00400002&tstat=000001011648&cycle=0&tclass1=000001138504&tclass2=000001138505).
#
# This is just an artificial data.
#
# 身長のデータは、[学校保健統計調査 令和元年度 全国表 | ファイル | 統計データを探す | 政府統計の総合窓口](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00400002&tstat=000001011648&cycle=0&tclass1=000001138504&tclass2=000001138505) のデータをもとに平均と標準偏差を計算し、再サンプルしています。
# なので、負の身長や、3m以上の身長があります。
#
# あくまで、人工データとしてお楽しみください。

# # overview

profile = ProfileReport(df_data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

profile

# # Qualitization I 

# ## preprocessing

y = df_data['身長']
X = pd.get_dummies(df_data[['学校', '性別']])[['学校_中', '学校_高', '性別_女']]

# The restriction of columns into `[['学校_中', '学校_高', '性別_女']]` is equivalent to set $a_\text{小} = a_\text{男} = 0$.

# ## apply Qualitization I (linear regression)

base_model = LinearRegression()
base_model.fit(X, y)

print('a:')
display(base_model.coef_)
print('b:')
display(base_model.intercept_)

df_data[(df_data['学校'] == '小') & (df_data['性別'] == '男')]['身長'].mean()

# Thus, we have the result of qualitization I as:
#
# $$
# \begin{align}
#     a_\text{小} &= 0 \\
#     a_\text{中} &= 24.54 \\
#     a_\text{高} &= 32.03 \\
#     a_\text{男} &= 0 \\
#     a_\text{女} &= -5.69 \\
#     b &= 134.07.
# \end{align}
# $$
#
# We can understand the following facts from the above result:
# - The average height of boys in elementary school is roughly 134.07cm. (Actually 132.16 cm. This is a numerical error. Theoretically, they must coincide.)
# - To belong junior-high school ("中学校" in Japanese) means that height is 24.54 cm higher. (The qualitization of "elementary school" is 24.54 cm.)
# - To be a female means that the height is 5.69 cm smaller. (The qualitization of "female" is -5.69 cm.)
# - etc.
#
# これは、数量化I類の結果として、
#
# $$
# \begin{align}
#     a_\text{小} &= 0 \\
#     a_\text{中} &= 24.54 \\
#     a_\text{高} &= 32.03 \\
#     a_\text{男} &= 0 \\
#     a_\text{女} &= -5.69 \\
#     b &= 134.07
# \end{align}
# $$
#
# という意味になる。
#
#
# これから、次の結果が読み取れる。
# - 小学生男子の平均身長は概ね 134.07cm。（実際は 132.16 cm。本来は一致するはずだが、数値的な誤差がある模様。）
# - 中学校に所属するということは、身長という尺度で言えば、24.54 cm 高いということを意味する。（「中学校」の数量化が 24.54 cm。）
# - 女性であるということは、身長という尺度で言えば、5.69 cm 小さいということを意味する。（「女性」の数量化が -5.69 cm。）
# - etc.

# ## Qualitization I relative to average
#
# Compute a and b of $\sum a_\text{小 or 中 or 大} = 0$ and $\sum a_\text{男 or 女} = 0$.
#
# $\sum a_\text{小 or 中 or 大} = 0$ かつ $\sum a_\text{男 or 女} = 0$ の場合の a, b を求める。

# +
X_only_school = X.copy()
X_only_school['性別_女'] = 0

bias_school = base_model.predict(X_only_school).mean() - base_model.intercept_
# -

# The value `bias_school` coincides with the average of $a_\text{小 or 中 or 大}$.

# +
X_only_sex = X.copy()
X_only_sex[['学校_中', '学校_高']] = 0

bias_sex = base_model.predict(X_only_sex).mean() - base_model.intercept_

# +
a_shou = - bias_school
a_chuu = base_model.coef_[0] - bias_school
a_kou = base_model.coef_[1] - bias_school

a_male = - bias_sex
a_female = base_model.coef_[2] - bias_sex

b = base_model.intercept_ + bias_school + bias_sex

print('a_shou')
print(a_shou)
print('a_chuu')
print(a_chuu)
print('a_kou')
print(a_kou)
print('a_male')
print(a_male)
print('a_female')
print(a_female)
print('b')
print(b)
# -

df_data['身長'].mean()

# Thus, we have the result of qualitization I as:
#
# $$
# \begin{align}
#     a_\text{小} &= -14.14 \\
#     a_\text{中} &= 10.40 \\
#     a_\text{高} &= 17.89 \\
#     a_\text{男} &= 2.85 \\
#     a_\text{女} &= -2.85 \\
#     b &= 145.36.
# \end{align}
# $$
#
# We can understand the following facts from the above result:
# - The average height is roughly 145.36cm. (actually, 145.36 cm.)
# - To belong elementary school ("小学校" in Japanese) means that height is 14.14 cm smaller. (The qualitization of "elementary school" is -14.14 cm.)
# - To be a female means that the height is 2.85 cm smaller. (The qualitization of "female" is -2.85 cm.)
# - etc.
#
#
# これは、数量化I類の結果として、
#
# $$
# \begin{align}
#     a_\text{小} &= -14.14 \\
#     a_\text{中} &= 10.40 \\
#     a_\text{高} &= 17.89 \\
#     a_\text{男} &= 2.85 \\
#     a_\text{女} &= -2.85 \\
#     b &= 145.36
# \end{align}
# $$
# という意味になる。
#
# これから、次の結果が読み取れる。
# - 平均身長は概ね 145.36cm。（実際 145.36 cm）
# - 小学校に所属するということは、身長という尺度で言えば、14.14 cm 小さいということを意味する。（「小学校」の数量化が -14.14 cm。）
# - 女性であるということは、身長という尺度で言えば、2.85 cm 小さいということを意味する。（「女性」の数量化が -2.85 cm。）
# - etc.


