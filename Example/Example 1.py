# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:00:15 2020

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

"""
This file is a simple demonstration of how to use the data interrogation/model 
construction/model evaluation files on your own and set hyperparameters. The data
file is the 3D example used in the paper.
"""

import numpy as np
import pandas as pd
from dataset_property_new import nonlinearity_assess, collinearity_assess,  residual_analysis
import cv_final as cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Load data
"""
Data = pd.read_excel('Printer.xlsx', header = None)     #d00.dat is for TEP  no delimiter #
Data = np.array(Data)

X = Data[:,0:9]

y = Data[:,10].reshape(-1,1)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state= 5)




"""
Data interrogation
"""
xticks = ['LH','WT','ID','IP','NT','BT','PS','M','FS']
yticks = ['TS']


round_number = 0
nonlinearity_assess(X, y, True, cat = [0,0,0,1,0,0,0,1,0] ,alpha = 0.01, difference = 0.4, xticks = xticks, yticks = yticks, round_number =  round_number)
collinearity_assess(X, y, True, xticks = xticks, yticks = yticks, round_number = round_number)



"""
Model construction and valuation
"""
#Based on the data interrogation result, the static nonlinear model is selected.
#Here several models are constructed to illustrate how to use the model construction.
#This file is just an illustration of how to use, but not the optimal way of model consturction for this example.

#ALVEN-------------------------------------------------------------------------
#The ALVEN model file takes unscaled data, and the final output prediction result is scaled
K_fold =5
Nr = 10
ALVEN_hyper,ALVEN_model, ALVEN_params, mse_train_ALVEN, mse_test_ALVEN, yhat_train, yhat_test, MSE_validation, final_list = cv.CV_mse('ALVEN', X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = K_fold, Nr= Nr, alpha_num=30, degree = [1,2], label_name=True)

scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y)
y_test = scaler_y.transform(y_test)

residual_analysis(X_test,y_test, yhat_test, alpha = 0.01, round_number = 'ALVEN')


#SVR---------------------------------------------------------------------------
#Except ALVEN/DALVEN, other models in SPA requires scaling
scaler_x = StandardScaler(with_mean=True, with_std=True)
scaler_x.fit(X)
X = scaler_x.transform(X)
X_test = scaler_x.transform(X_test)
    
scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y)
y = scaler_y.transform(y)
y_test = scaler_y.transform(y_test) 


K_fold = 5
Nr = 10
SVR_hyper, SVR_model, mse_train_SVR, mse_test_SVR, yhat_train, yhat_test, MSE_validate = cv.CV_mse('SVR', X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = K_fold, Nr= Nr, C = [0.001, 1 , 100])

y_test = scaler_y.inverse_transform(y_test)
yhat_test = scaler_y.inverse_transform(yhat_test)


residual_analysis(X_test,y_test, yhat_test, alpha = 0.01, round_number = 'SVR')
