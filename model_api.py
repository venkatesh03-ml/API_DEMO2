# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:03:37 2020

@author: z016835
"""

import pandas as pd
import numpy as np

# Load the dataset in a dataframe object and include only four features as mentioned
url = "C:\\Users\\z016835\\Desktop\\case study\\api\\train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df_ = df[include]

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)


from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

