# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:35:56 2020

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
import regression_models as rm
from sklearn.utils import resample
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import nonlinear_regression as nr
from sklearn.model_selection import train_test_split
from copy import deepcopy
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import nonlinear_regression_other as nro
from sklearn.feature_selection import VarianceThreshold



def CVpartition(X, y, Type = 'Re_KFold', K = 10, Nr = 1000, random_state = 0, group = None):
    '''This function create partition for data for cross validation and bootstrap
    https://scikit-learn.org/stable/modules/cross_validation.html
    
    Input:
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    type: 'KFold', 'Re_KFold', 'MC' for cross validation
          'TS' for time series cross validation
    group: group index for grouped CV
          
    K: float, 1/K portion of data will be used as validation set, default 10
    Output:partitioned data set
    Nr: Number of repetitions, ignored whtn CV_type = 'KFold', for Re_KFold, it will be Nr * K total
    
    Output:generator (X_train, y_train, X_val, y_val)
    '''
    
    if Type == 'MC':
        CV = ShuffleSplit(n_splits=Nr, test_size=1/K, random_state =random_state)
        for train_index, val_index in CV.split(X,y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
            
    elif Type == 'Single':
        X, X_test, y, y_test = train_test_split(X, y, test_size=1/K, random_state =random_state)
        yield (X, y ,X_test, y_test)

    elif Type == 'KFold':
        CV = KFold(n_splits = int(K), random_state =random_state)
        for train_index, val_index in CV.split(X,y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
        
    elif Type == 'Re_KFold':
        CV = RepeatedKFold(n_splits= int(K), n_repeats= Nr, random_state =random_state)
        for train_index, val_index in CV.split(X,y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
            
    elif Type == 'Timeseries':
        TS = TimeSeriesSplit(n_splits=int(K))
        for train_index, val_index in TS.split(X):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    
    
    elif Type == 'Group':
        #the repliates should have the same group indicator
        #the size of indicator indicates the range of DOE
        label =np.unique(group)
        print('***** '+str(len(label))+' fold is used for CV as default ******')
        for i in range(len(label)):
            yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])],y[np.squeeze(group == label[i])])

    elif Type == 'Group_no_extrapolation':
        #for no extrapolation case, 
        label =np.unique(group)
        print('***** '+str(len(label)-2)+' fold is used for CV as default ******')
        for i in range(len(label)):
            if min(label)<label[i] and label[i]<max(label):
                yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])],y[np.squeeze(group == label[i])])
   
    elif Type == 'GroupKFold':
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits = int(K))
        for train_index, val_index in gkf.split(X, y, groups=group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    

    elif Type == 'GroupShuffleSplit':
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits = int(Nr), test_size = 1/K, random_state=random_state)
        for train_index, val_index in gss.split(X, y, groups=label):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    
  
    elif Type == 'No_CV':
        yield (X,y,X,y)
    
    elif Type == 'Single_ordered':
        num = X.shape[0]
        yield (X[:num-round(X.shape[0]*1/K):], y[:num-round(X.shape[0]*1/K):] , X[num-round(X.shape[0]*1/K):], y[num-round(X.shape[0]*1/K):])        
                
    else:
        print('Wrong type specified for data partition')



def CV_mse(model_name, X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = 10, Nr = 1000, eps = 1e-4,alpha_num=50, group = None, round_number = '', **kwargs):
    '''This function determines the best hyper_parameter using mse based on CV
    Input:
    model_name: str, indicating which model to use
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    cv_type: cross_validation type
    K: fold for CV
    Nr: repetition for CV
    **kwargs: hyper-parameters for model fitting, if None, using default range or settings
    
    
    Output: 
    hyper_params: dictionary, contains optimal model parameters based on cross validation
    model: final fited model on all training data
    model_params: np_array m x 1
    mse_train
    mse_test
    yhat_train
    yhat_test
    '''
    
    if model_name == 'EN':
        EN = rm.model_getter(model_name)
        

        if 'l1_ratio' not in kwargs:
            #kwargs['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50]
            kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
            
        MSE_result = np.zeros((alpha_num,len(kwargs['l1_ratio']),1))
        Var = np.zeros((alpha_num,len(kwargs['l1_ratio']), 1))  #alpha, l1, counter

        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            if counter >1:
                MSE_result = np.c_[MSE_result, np.zeros((alpha_num,len(kwargs['l1_ratio']),1))]
                Var = np.c_[Var, np.zeros((alpha_num,len(kwargs['l1_ratio']), 1)) ]
            for j in range(len(kwargs['l1_ratio'])):
                if kwargs['l1_ratio'][j] == 0:
                    alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
                    kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]

                    for i in range(alpha_num):
                        clf = Ridge(alpha=kwargs['alpha'][i],fit_intercept=False).fit(X_train, y_train)
                        mse = np.sum((clf.predict(X_val)-y_val)**2)/y_val.shape[0]                                                
                        variable = clf.coef_
                        MSE_result[i,j, counter-1] = mse
                        Var[i,j,counter-1] = np.sum(variable.flatten() != 0)
                else:
                    alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/kwargs['l1_ratio'][j]
                    kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]
                    for i in range(alpha_num):
                        _, variable, _, mse, _, _ = EN(X_train, y_train, X_val, y_val, alpha = kwargs['alpha'][i], l1_ratio = kwargs['l1_ratio'][j])

                        MSE_result[i,j, counter-1] = mse
                        Var[i,j,counter-1] = np.sum(variable.flatten() != 0)
                
        MSE_mean = np.sum(MSE_result, axis = 2)/counter
        MSE_std = np.std(MSE_result, axis = 2)
        Var_num = np.sum(Var,axis=2) #avegae over K*Nr
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)
        

        MSE_min = MSE_mean[ind[0],ind[1]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1]]        
        
        #find the one that right below/ith the smallest number of variables
        Var_num_final = Var_num[MSE_mean<MSE_bar].min()
        ind_num = Var_num == Var_num_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        #find the min value, if there is a tie, only the first occurence is returned
        l1_ratio = kwargs['l1_ratio'][int(final_ind[1][0])]
            
        if l1_ratio != 0:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/l1_ratio
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]
           
            alpha = kwargs['alpha'][int(final_ind[0][0])]
        else:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]
           
            alpha = kwargs['alpha'][int(final_ind[0][0])]
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        
        #fit the final model using opt hyper_params
        if l1_ratio == 0:
            EN_model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
            EN_params= EN_model.coef_.reshape(-1,1)
            yhat_train = EN_model.predict(X)
            yhat_test = EN_model.predict(X_test)
            mse_train = np.sum((yhat_train-y)**2)/y.shape[0]  
            mse_test = np.sum((yhat_test-y_test)**2)/y_test.shape[0]  
        else:
            EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test = EN(X, y, X_test, y_test, alpha = alpha, l1_ratio = l1_ratio)
        
        return(hyper_params, EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[final_ind[0][0],final_ind[1][0]])
    ########################################################################################################################################   
        
    elif model_name == 'SPLS':
        SPLS = rm.model_getter(model_name)
        
        if cv_type != 'Group' and cv_type != 'Group_no_extrapolation' and cv_type != 'GroupKFold' and cv_type != 'GroupShuffleSplit':
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)),min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)))
            if 'eta' not in kwargs:
                kwargs['eta'] = np.linspace(0,1,20, endpoint = False)[::-1] #eta = 0 use normal PLS
           
            MSE_result = np.zeros((len(kwargs['K']),len(kwargs['eta']),1))
            Var = np.zeros((len(kwargs['K']),len(kwargs['eta']), 1))  #K, eta, counte
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
                counter += 1
                if counter >1:
                    MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['K']),len(kwargs['eta']),1))]
                    Var = np.c_[Var, np.zeros((len(kwargs['K']),len(kwargs['eta']), 1))]
 
                for i in range(len(kwargs['K'])):
                    for j in range(len(kwargs['eta'])):
                        _, variable, _, mse, _, _ = SPLS(X_train, y_train, X_val, y_val, K = int(kwargs['K'][i]), eta = kwargs['eta'][j], eps = eps)
                        MSE_result[i,j, counter-1] = mse
                        Var[i,j,counter-1] = np.sum(variable.flatten() != 0)
                
        
        else:
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int(X.shape[0]-1)),min(X.shape[1],int(X.shape[0]-1)))
            if 'eta' not in kwargs:
                kwargs['eta'] = np.linspace(0,1,20, endpoint = False)[::-1] #eta = 0 use normal PLS
           
            MSE_result = np.zeros((len(kwargs['K']),len(kwargs['eta']),1))
            Var = np.zeros((len(kwargs['K']),len(kwargs['eta']), 1))  #K, eta, counter
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
                counter += 1
                if counter >1:
                    MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['K']),len(kwargs['eta']),1))]
                    Var = np.c_[Var, np.zeros((len(kwargs['K']),len(kwargs['eta']), 1))]
                    
                for i in range(len(kwargs['K'])):
                    for j in range(len(kwargs['eta'])):
                        if kwargs['K'][i] > X_train.shape[0]-1:
                            mse = 10000
                            variable = np.ones((X_train.shape[1],1))
                            ###############################################should be removed in the future
                        elif kwargs['K'][i] > 30:
                            mse = 10000
                            variable = np.ones((X_train.shape[1],1))
                        else:
                            _, variable, _, mse, _, _ = SPLS(X_train, y_train, X_val, y_val, K = int(kwargs['K'][i]), eta = kwargs['eta'][j], eps = eps)
                        
                        MSE_result[i,j, counter-1] = mse
                        Var[i,j,counter-1] = np.sum(variable.flatten() != 0)
                
                    
        MSE_mean = np.sum(MSE_result, axis = 2)/counter
        MSE_std = np.std(MSE_result, axis = 2)
        Var_num = np.sum(Var,axis=2) #avegae over K*Nr
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)                    
            
        
        MSE_min = MSE_mean[ind[0],ind[1]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1]]        
        
        #find the one that right below/ith the smallest number of variables
        Var_num_final = Var_num[MSE_mean<MSE_bar].min()
        ind_num = Var_num == Var_num_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        #find the min value, if there is a tie, only the first occurence is returned
        K = kwargs['K'][int(final_ind[0][0])]
        eta = kwargs['eta'][int(final_ind[1][0])]
            
        hyper_params = {}
        hyper_params['K'] = int(K)
        hyper_params['eta'] = eta
                
        
        #fit the final model using opt hyper_params
        SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test = SPLS(X, y, X_test, y_test, eta = eta, K = K)
        
        return(hyper_params, SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[final_ind[0][0],final_ind[1][0]])
    
    ########################################################################################################################################   
             
    elif model_name == 'PLS':
        
        if cv_type != 'Group' and cv_type != 'Group_no_extrapolation' and cv_type != 'GroupKFold' and cv_type != 'GroupShuffleSplit':
          
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)),min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)))
           
            MSE_result = np.zeros((len(kwargs['K']),1))

            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group):
                counter += 1
                if counter >1:
                    MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['K']),1))]
                for i in range(len(kwargs['K'])):
                    PLS = PLSRegression(scale = False, n_components=int(kwargs['K'][i]), tol = eps).fit(X_train,y_train)
                    PLS_para = PLS.coef_.reshape(-1,1)
                    yhat = np.dot(X_val, PLS_para)
                    MSE_result[i,counter-1] = rm.mse(y_val, yhat)
        
        else:
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int(X.shape[0]-1)),min(X.shape[1],int(X.shape[0]-1)))
                     
            MSE_result = np.zeros((len(kwargs['K']),1))
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group):
                counter += 1
                if counter >1:
                    MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['K']),1))]
                for i in range(len(kwargs['K'])):
                    if kwargs['K'][i] > X_train.shape[0]-1:
                        MSE_result[i,counter-1] = 10000
                    else:                        
                        PLS = PLSRegression(scale = False, n_components=int(kwargs['K'][i]), tol = eps).fit(X_train,y_train)
                        PLS_para = PLS.coef_.reshape(-1,1)
                        yhat = np.dot(X_val, PLS_para)
                        MSE_result[i,counter-1] = rm.mse(y_val, yhat)
                              
                            
        MSE_mean = np.sum(MSE_result, axis = 1)/counter
        MSE_std = np.std(MSE_result, axis = 1)
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)
        
        MSE_min = MSE_mean[ind[0]]
        MSE_bar = MSE_min + MSE_std[ind[0]]
               
        #find the min number of K within one std of min CV error
        hyper_params = {}
        hyper_params['K'] = int(kwargs['K'][MSE_mean<MSE_bar].min())
        index_final = np.where(kwargs['K']==hyper_params['K'])     
        
        #fit the final model using opt hyper_params
        PLS_model = PLSRegression(scale = False, n_components=int(hyper_params['K'])).fit(X,y)
        PLS_params = PLS_model.coef_.reshape(-1,1)
        
        yhat_train = np.dot(X, PLS_params)
        yhat_test = np.dot(X_test, PLS_params)
        
        mse_train = rm.mse(yhat_train,y)
        mse_test = rm.mse(yhat_test,y_test)
        
        return(hyper_params, PLS_model, PLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[index_final][0])
     


     ########################################################################################################################################
         
    elif model_name == 'RR':
        
        if 'alpha' not in kwargs:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]
   
        MSE_result = np.zeros((len(kwargs['alpha']),1))
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter+=1
            if counter >1:
                MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['alpha']),1))]            
            for i in range(len(kwargs['alpha'])):
                RR = Ridge(alpha = kwargs['alpha'][i], fit_intercept = False).fit(X_train, y_train)
                Para = RR.coef_.reshape(-1,1)
                yhat = np.dot(X_val, Para)
                MSE_result[i,counter-1] = rm.mse(y_val, yhat)
                                        
            
        #find the min value, if there is a tie, only the first occurence is returned
        MSE_mean = np.sum(MSE_result, axis = 1)/counter
        MSE_std = np.std(MSE_result, axis = 1)
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)
        
        MSE_min = MSE_mean[ind[0]]
        MSE_bar = MSE_min + MSE_std[ind[0]]
               
        #find the min number of K within one std of min CV error
        hyper_params = {}
        hyper_params['alpha'] = kwargs['alpha'][MSE_mean<MSE_bar].max()
        index_final = np.where(kwargs['alpha']==hyper_params['alpha'])     
                
        
        
        #fit the final model using opt hyper_params
        RR_model = Ridge(alpha = hyper_params['alpha'], fit_intercept = False).fit(X,y)
        RR_params = RR_model.coef_.reshape(-1,1)
        
        yhat_train = np.dot(X, RR_params)
        yhat_test = np.dot(X_test, RR_params)
        
        mse_train = rm.mse(yhat_train,y)
        mse_test = rm.mse(yhat_test,y_test)
        
        return(hyper_params, RR_model, RR_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[index_final][0])
 
    ########################################################################################################################################   
    
    if model_name == 'ALVEN':
        ALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
            
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['ALVEN_select_pvalue'] = 0.15



            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']),1))
        Var = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']),1))

        
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            if counter > 1:
                MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']),1))]
                Var = np.c_[Var, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']),1))]

            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        _, variable,_, mse, _, _ , _, _= ALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['ALVEN_select_pvalue'], trans_type = kwargs['trans_type'])
                        
                        MSE_result[k,i,j, counter - 1] = mse
                        Var[k,i,j, counter - 1] = np.sum(variable.flatten() !=0)

        MSE_mean = np.sum(MSE_result, axis = 3)/counter
        MSE_std = np.std(MSE_result, axis = 3)
        Var_num = np.sum(Var,axis=3) 
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)

        MSE_min = MSE_mean[ind[0],ind[1],ind[2]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2]]        
        
        #find the one that right below/ith the smallest number of variables
        Var_num_final = Var_num[MSE_mean<MSE_bar].min()
        ind_num = Var_num == Var_num_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        degree = kwargs['degree'][int(final_ind[0][0])]
        l1_ratio = kwargs['l1_ratio'][int(final_ind[2][0])]
       
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index= ALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['ALVEN_select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        
        
        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            Xtrans,_ = nr.feature_trans(X, degree = degree, interaction = 'later')
        else:
            Xtrans, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)

        sel = VarianceThreshold(threshold=eps).fit(Xtrans)
        
        

        if kwargs['label_name'] :
            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]
                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(X.shape[1]-2):
                        for j in range(i+1,X.shape[1]-1):
                            for k in range(j+1,X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]
            else:
                if degree == 1:
                    list_name_final = list_name
                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final= [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)
     ########################################################################################################################################   
         
    elif model_name == 'RF':
        
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = [3,5,10,15,20,40]
        if 'n_estimators' not in kwargs:
            kwargs['n_estimators'] = [10, 50, 100, 200]
        if 'min_samples_leaf' not in kwargs:
            kwargs['min_samples_leaf'] = [0.005, 0.01, 0.05, 0.1] #, 0.05 ,0.1, 0.2] # 0.3, 0.4]
        
        MSE_result = np.zeros((len(kwargs['max_depth']),len(kwargs['n_estimators']),len(kwargs['min_samples_leaf']),1))

        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            if counter >1:
                 MSE_result = np.c_[MSE_result,np.zeros((len(kwargs['max_depth']),len(kwargs['n_estimators']),len(kwargs['min_samples_leaf']),1))]

            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        _, _, mse, _,_ = nro.RF_fitting(X_train, y_train, X_val,y_val, n_estimators = kwargs['n_estimators'][j], max_depth = kwargs['max_depth'][i], min_samples_leaf=kwargs['min_samples_leaf'][k])
                        MSE_result[i,j,k,counter-1] = mse
        
        #create score matrix
        S = np.zeros((len(kwargs['max_depth']),len(kwargs['n_estimators']),len(kwargs['min_samples_leaf'])))
        
        for i in range(len(kwargs['max_depth'])):
            for j in range(len(kwargs['n_estimators'])):
                for k in range(len(kwargs['min_samples_leaf'])):
                    S[i,j,k] = i/len(kwargs['max_depth'])-k/len(kwargs['min_samples_leaf'])

            
        #find the min value, if there is a tie, only the first occurence is returned
        MSE_mean = np.sum(MSE_result, axis = 3)/counter
        MSE_std = np.std(MSE_result, axis = 3)       
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)                    

        MSE_min = MSE_mean[ind[0],ind[1],ind[2]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2]]  
        
        #find thw one that right below the bar with minmum_depth and max_end_leaves
        S_final = S[MSE_mean<MSE_bar].min()
        ind_num = S == S_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)

        #find the min value, if there is a tie, only the first occurence is returned
        max_depth = kwargs['max_depth'][int(final_ind[0][0])]
        n_estimators = kwargs['n_estimators'][int(final_ind[1][0])]
        min_samples_leaf = kwargs['min_samples_leaf'][int(final_ind[2][0])]
            
        hyper_params = {}
        hyper_params['max_depth'] = max_depth
        hyper_params['n_estimators'] = n_estimators
        hyper_params['min_samples_leaf'] = min_samples_leaf
        
        #fit the final model using opt hyper_params
        RF_model, mse_train, mse_test, yhat_train, yhat_test = nro.RF_fitting(X, y, X_test, y_test, n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        
        return(hyper_params, RF_model, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
 
    ########################################################################################################################################   
  
    elif model_name == 'SVR':
        
        if 'C' not in kwargs:
            kwargs['C'] = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        if 'gamma' not in kwargs:
            gd = 1/X.shape[1]
            kwargs['gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3]
        
        MSE_result = np.zeros((len(kwargs['C']),len(kwargs['gamma']),len(kwargs['epsilon']),1))
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            if counter > 1:
                MSE_result = np.c_[MSE_result,np.zeros((len(kwargs['C']),len(kwargs['gamma']),len(kwargs['epsilon']),1))]
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        _, _, mse, _,_ = nro.SVR_fitting(X_train, y_train, X_val,y_val,
                                                     C=kwargs['C'][i], gamma=kwargs['gamma'][j], epsilon=kwargs['epsilon'][k])
                        MSE_result[i,j,k,counter-1] = mse
                                        
        #score matrix
        S= np.zeros((len(kwargs['C']),len(kwargs['gamma']),len(kwargs['epsilon'])))
        for i in range(len(kwargs['C'])):
            for j in range(len(kwargs['gamma'])):
                for k in range(len(kwargs['epsilon'])):
                    S[i,j,k] = i/len(kwargs['C'])-j/len(kwargs['gamma'])-k/len(kwargs['epsilon'])
        
        #find the min value, if there is a tie, only the first occurence is returned
        MSE_mean = np.sum(MSE_result, axis = 3)/counter
        MSE_std = np.std(MSE_result, axis = 3)       
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)                    

        MSE_min = MSE_mean[ind[0],ind[1],ind[2]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2]]  
        
        #find thw one that right below the bar with minmum_depth and max_end_leaves
        S_final = S[MSE_mean<MSE_bar].min()
        ind_num = S == S_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        C = kwargs['C'][int(final_ind[0][0])]
        gamma = kwargs['gamma'][int(final_ind[1][0])]
        epsilon = kwargs['epsilon'][int(final_ind[2][0])]

        hyper_params = {}
        hyper_params['C'] = C
        hyper_params['gamma'] = gamma
        hyper_params['epsilon'] = epsilon
        
        #fit the final model using opt hyper_params
        SVR_model, mse_train, mse_test, yhat_train, yhat_test =  nro.SVR_fitting(X, y, X_test,y_test,
                                                                                C=C, gamma=gamma, epsilon=epsilon)
        
        return(hyper_params, SVR_model, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
 
    ########################################################################################################################################   
  
   
      
    elif model_name == 'DALVEN':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] = [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
            
            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))
        Var = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))

        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1

            if counter > 1:
                MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))]   
                Var = np.c_[Var, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))]   
       
            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        for t in range(len(kwargs['lag'])):
                            _, variable,_ , mse, _, _ , _, _,_= DALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                            MSE_result[k,i,j,t,counter - 1] = mse
                            Var[k,i,j,t,counter-1] = np.sum(variable.flatten() !=0)

        MSE_mean = np.sum(MSE_result, axis = 4)/counter
        MSE_std = np.std(MSE_result, axis = 4)
        Var_num = np.sum(Var,axis=4) 
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)

        MSE_min = MSE_mean[ind[0],ind[1],ind[2],ind[3]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2],ind[3]]  


        #find the one that right below/ith the smallest number of variables
        Var_num_final = Var_num[MSE_mean<MSE_bar].min()
        ind_num = Var_num == Var_num_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        
        degree = kwargs['degree'][int(final_ind[0][0])]
        l1_ratio = kwargs['l1_ratio'][int(final_ind[2][0])]
        lag = kwargs['lag'][int(final_ind[3][0])]
       
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,_= DALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, lag = lag, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        hyper_params['lag'] = lag
        hyper_params['retain_index'] = retain_index

        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            Xtrans,_ = nr.feature_trans(X, degree = degree, interaction = 'later')
        else:
            Xtrans, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)


    
        #lag padding for X
        XD = Xtrans[lag:]
        for i in range(lag):
            XD = np.hstack((XD,Xtrans[lag-1-i:-i-1]))
            
        #lag padding for y in design matrix
        for i in range(lag):
            XD = np.hstack((XD,y[lag-1-i:-i-1]))
        
        #remove feature with 0 variance
        sel = VarianceThreshold(threshold=eps).fit(XD)



        if kwargs['label_name'] :
            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                    
                      
                        
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        


                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(X.shape[1]-2):
                        for j in range(i+1,X.shape[1]-1):
                            for k in range(j+1,X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]

                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        




            else:
                if degree == 1:
                    list_name_final = list_name
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                  
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)
     



    ########################################################################################################################################   
      
    elif model_name == 'DALVEN_full_nonlinear':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2] #,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] = [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
            
            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))
        Var = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1

            if counter > 1:
                MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))]   
                Var = np.c_[Var, np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag']),1))]   
       
            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        for t in range(len(kwargs['lag'])):
                            _, variable, _, mse, _, _ , _, _,_= DALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                            MSE_result[k,i,j,t,counter - 1] = mse
                            Var[k,i,j,t,counter-1] = np.sum(variable.flatten() !=0)

            
        MSE_mean = np.sum(MSE_result, axis = 4)/counter
        MSE_std = np.std(MSE_result, axis = 4)
        Var_num = np.sum(Var,axis=4) 
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)

        MSE_min = MSE_mean[ind[0],ind[1],ind[2],ind[3]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2],ind[3]]  

            
        #find the one that right below/ith the smallest number of variables
        Var_num_final = Var_num[MSE_mean<MSE_bar].min()
        ind_num = Var_num == Var_num_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
        
        degree = kwargs['degree'][int(final_ind[0][0])]
        l1_ratio = kwargs['l1_ratio'][int(final_ind[2][0])]
        lag = kwargs['lag'][int(final_ind[3][0])]
        
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,_= DALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, lag = lag, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        hyper_params['lag'] = lag
        hyper_params['retain_index'] = retain_index



        #lag padding for X
        XD = X[lag:]
        for i in range(lag):
            XD = np.hstack((XD,X[lag-1-i:-i-1]))
            
        #lag padding for y in design matrix
        for i in range(lag):
            XD = np.hstack((XD,y[lag-1-i:-i-1]))
            
        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            XD,_ = nr.feature_trans(XD, degree = degree, interaction = 'later')
        else:
            XD, _ = nr.poly_feature(XD, degree = degree, interaction = True, power = True)

        
        #remove feature with 0 variance
        sel = VarianceThreshold(threshold=eps).fit(XD)



        if kwargs['label_name'] :
            list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            list_copy = list_name[:]

            for i in range(lag):
                list_name = list_name + [s + '(t-' + str(i+1) + ')' for s in list_copy]
            for i in range(lag):
                list_name = list_name + ['y(t-' + str(i+1) +')' ] 
                    
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                 
                        
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(len(list_name)-1):
                        for j in range(i+1,len(list_name)):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]


                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(len(list_name)-1):
                        for j in range(i+1,len(list_name)):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(len(list_name)-2):
                        for j in range(i+1,len(list_name)-1):
                            for k in range(j+1,len(list_name)):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]                    




            else:
                if degree == 1:
                    list_name_final = list_name
                    

                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
       

                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(len(list_name)):
                        for j in range(i, len(list_name)):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(len(list_name)):
                        for j in range(i, len(list_name)):
                            for k in range(j, len(list_name)):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                  

                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)



    ########################################################################################################################################   
    #for RNN, only the model archetecture is viewed as hyper-parameter in thie automated version, the other training parameters can be set by kwargs, otw the default value will be used


    elif model_name == 'RNN':
        import timeseries_regression_RNN as RNN

        input_size_x = X.shape[1]
        
        #currently not support BRNN version, so keep test at 1
        input_prob_test = 1
        output_prob_test = 1
        state_prob_test = 1
        
        #model architecture (which is also hyperparameter for selection)
        if 'cell_type' not in kwargs:
            kwargs['cell_type'] = 'e'
        if 'activation' not in kwargs:
            kwargs['activation'] = ['tanh'] #can be relu, tanh,  linear
        if 'state_size' not in kwargs:
            kwargs['state_size'] = [input_size_x*(i+1) for i in range(5)]
        if 'num_layers' not in kwargs:
            kwargs['num_layers'] = [1, 2, 3]
        

        #training parameters
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 2
        if 'epoch_overlap' not in kwargs:
            kwargs['epoch_overlap'] = 0
        if 'num_steps' not in kwargs:
            kwargs['num_steps'] = 25
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 1e-2
        if 'lambda_l2_reg' not in kwargs:
            kwargs['lambda_l2_reg'] = 1e-3
        if 'num_epochs' not in kwargs:
            kwargs['num_epochs'] = 1000
  
       #drop-out parameters for training

        if 'input_prob' not in kwargs:
            kwargs['input_prob'] = 1
        if 'output_prob' not in kwargs:
            kwargs['output_prob'] = 1
        if 'state_prob' not in kwargs:
            kwargs['state_prob'] = 1
        

        #early stop parameter
        if 'train_ratio' not in kwargs:
            kwargs['train_ratio'] = 0.85

        if 'max_checks_without_progress' not in kwargs:
            kwargs['max_checks_without_progress'] = 100
        if 'epoch_before_val' not in kwargs:
            kwargs['epoch_before_val'] = 300
        
        
        #save or not
        if 'location' not in kwargs:
            kwargs['location'] = 'RNNtest'
        

                
                
        MSE_result = np.zeros((len(kwargs['cell_type']),len(kwargs['activation']), len(kwargs['state_size']), len(kwargs['num_layers']),1))
        S = np.zeros((len(kwargs['cell_type']),len(kwargs['activation']), len(kwargs['state_size']), len(kwargs['num_layers']),1))
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            
            if counter > 1:
                MSE_result = np.c_[MSE_result, np.zeros((len(kwargs['cell_type']),len(kwargs['activation']), len(kwargs['state_size']), len(kwargs['num_layers']),1))]   
                S = np.c_[S, np.zeros((len(kwargs['cell_type']),len(kwargs['activation']), len(kwargs['state_size']), len(kwargs['num_layers']),1))]   
       
    
            for i in range(len(kwargs['cell_type'])):
                for j in range(len(kwargs['activation'])):
                    for k in range(len(kwargs['state_size'])):
                        for t in range(len(kwargs['num_layers'])):
                            _,_, _, _, _, _, test_loss = RNN.timeseries_RNN_feedback_single_train(X_train, y_train, X_test=X_val, Y_test=y_val, train_ratio = kwargs['train_ratio'],\
                                                                                             cell_type=kwargs['cell_type'][i],activation = kwargs['activation'][j], state_size = kwargs['state_size'][k],\
                                                                                             batch_size = kwargs['batch_size'], epoch_overlap = kwargs['epoch_overlap'],num_steps = kwargs['num_steps'],\
                                                                                             num_layers = kwargs['num_layers'][t], learning_rate = kwargs['learning_rate'],  lambda_l2_reg=kwargs['lambda_l2_reg'],\
                                                                                             num_epochs = kwargs['num_epochs'], input_prob = kwargs['input_prob'], output_prob = kwargs['output_prob'], state_prob = kwargs['state_prob'],\
                                                                                             input_prob_test =input_prob_test, output_prob_test = output_prob_test, state_prob_test =state_prob_test,\
                                                                                             max_checks_without_progress = kwargs['max_checks_without_progress'],epoch_before_val=kwargs['epoch_before_val'], location= kwargs['location'], plot= False)
                            
                            MSE_result[i,j,k,t,counter - 1] = test_loss
                            S[i,j,k,t,counter-1] = k*t+i+j

            
        MSE_mean = np.sum(MSE_result, axis = 4)/counter
        MSE_std = np.std(MSE_result, axis = 4)
        S_value = np.sum(S,axis=4) 
        ind = np.unravel_index(MSE_mean.argmin(), MSE_mean.shape)

        MSE_min = MSE_mean[ind[0],ind[1],ind[2],ind[3]]
        MSE_bar = MSE_min + MSE_std[ind[0],ind[1],ind[2],ind[3]]  

        
        print(MSE_mean)
        print(MSE_bar)
        #find the one that right below/ith the smallest number of variables
        S_value_final = S_value[MSE_mean<MSE_bar].min()
        ind_num = S_value == S_value_final
        ind_mse = MSE_mean<MSE_bar
        final_ind = np.where(ind_num*ind_mse)
        
                                    
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        cell_type = kwargs['cell_type'][int(final_ind[0][0])]
        activation = kwargs['activation'][int(final_ind[1][0])]
        state_size = kwargs['state_size'][int(final_ind[2][0])]
        num_layers = kwargs['num_layers'][int(final_ind[3][0])]
       
        print('Final training')
        prediction_train,prediction_val, prediction_test, _, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, X_test=X_test, Y_test=y_test, train_ratio = kwargs['train_ratio'],\
                                                                                         cell_type=cell_type,activation = activation , state_size = state_size,\
                                                                                         batch_size = kwargs['batch_size'], epoch_overlap = kwargs['epoch_overlap'],num_steps = kwargs['num_steps'],\
                                                                                         num_layers = num_layers, learning_rate = kwargs['learning_rate'],  lambda_l2_reg=kwargs['lambda_l2_reg'],\
                                                                                         num_epochs = kwargs['num_epochs'], input_prob = kwargs['input_prob'], output_prob = kwargs['output_prob'], state_prob = kwargs['state_prob'],\
                                                                                         input_prob_test =input_prob_test, output_prob_test = output_prob_test, state_prob_test =state_prob_test,\
                                                                                         max_checks_without_progress = kwargs['max_checks_without_progress'],epoch_before_val=kwargs['epoch_before_val'], location= kwargs['location'], plot= True,round_number = round_number)

        
        hyper_params = {}
        hyper_params['cell_type'] = cell_type
        hyper_params['activation'] = activation
        hyper_params['state_size'] = state_size
        hyper_params['num_layers'] = num_layers
        hyper_params['training_params'] = {'batch_size':kwargs['batch_size'],'epoch_overlap':kwargs['epoch_overlap'],'num_steps':kwargs['num_steps'],'learning_rate':kwargs['learning_rate'],'lambda_l2_reg':kwargs['lambda_l2_reg'],'num_epochs':kwargs['num_epochs']}
        hyper_params['drop_out'] = {'input_prob':kwargs['input_prob'],'output_prob':kwargs['output_prob'], 'state_prob':kwargs['state_prob']}
        hyper_params['early_stop'] = {'train_ratio':kwargs['train_ratio'], 'max_checks_without_progress':kwargs['max_checks_without_progress'],'epoch_before_val':kwargs['epoch_before_val']}
        hyper_params['MSE_val'] = MSE_result[ind]
        
        
        return(hyper_params, kwargs['location'], prediction_train, prediction_val, prediction_test, train_loss_final, val_loss_final, test_loss_final)



