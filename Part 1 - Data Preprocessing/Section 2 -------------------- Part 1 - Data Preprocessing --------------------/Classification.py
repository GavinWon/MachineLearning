# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:23:32 2020

@author: Gavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datasetTest = pd.read_csv('AW_test.csv')
datasetWorkCusts = pd.read_csv("AdvWorksCusts.csv")
datasetBikeBuy = pd.read_csv("AW_BikeBuyer.csv")

#Data preparation --> removing duplicates
datasetWorkCusts.columns = [str.replace('-', '_') for str in datasetWorkCusts.columns]
datasetWorkCusts.drop_duplicates(subset = 'CustomerID', keep = 'first', inplace = True)
print(datasetWorkCusts.shape)
print(datasetWorkCusts.CustomerID.unique().shape)

X_Train = datasetWorkCusts.iloc[:, 4:].values
Y_Train = datasetBikeBuy.iloc[:,1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_Train = np.array(ct.fit_transform(X_Train), dtype=np.float)





