# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:13:30 2018

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

import statsmodels.api as sm
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
from SPLS import SPLS_fitting_method
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.linear_model import Lasso
import nonlinear_regression as nr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
import math
import numpy.matlib as matlib
from sklearn.feature_selection import VarianceThreshold
#from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


def model_getter(model_name):
    '''Return the model according to the name'''
    
    switcher = {
            'OLS': OLS_fitting,
            'SPLS': SPLS_fitting,
            'EN': EN_fitting,
            'LASSO': LASSO_fitting,
            'ALVEN': ALVEN_fitting,
            'RR': RR_fitting,
            'DALVEN': DALVEN_fitting,
            'DALVEN_full_nonlinear': DALVEN_fitting_full_nonlinear
            }
    
    #Get the function from switcher dictionary
    if model_name not in switcher:
        print('No corresponding regression model')
    func = switcher.get(model_name)
    return func



def mse(y, yhat):
    """
    This function calculate the goodness of fit mse
    Input: y: N x 1 real response
           yhat: N x 1 predited by the model
           
    Output: mse
    """
    return np.sum((yhat-y)**2)/y.shape[0]  


def OLS_fitting(X, y, X_test, y_test, prob = False, alpha = 0.01):
    '''OLS fitting: y=ax1+bx2+...
    Input:
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    X_test: independent variables of size N_test x m np_array
    prob: Bool, whether to report confidence interval
    alpha: Significance level, default 0.01
    
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: stat_ols model type
    model_params: np_array m x 1
    '''

    #training
    OLS_model = sm.OLS(y, X).fit()
    yhat_train = OLS_model.predict().reshape((-1,1))
    mse_train = mse(y, yhat_train)
    
    ##getting AIC
    #AIC = OLS_model.aic
    
    #getting fitted parameters
    OLS_params = OLS_model.params.reshape(-1,1)
    
    #testing
    yhat_test = OLS_model.predict(X_test).reshape((-1,1))
    mse_test = mse(y_test, yhat_test)

    #confidence interval
    #each of size N x 1 array
    #if prob:
        #_, iv_l_train, iv_u_train = wls_prediction_std(OLS_model)
        #iv_l_train = iv_l_train.reshape(-1,1)
        #iv_u_train = iv_u_train.reshape(-1,1)
        #OLS_predictions = OLS_model.get_prediction(X_test)
        #iv_l_test = OLS_predictions.summary_frame(alpha = alpha).obs_ci_lower.reshape(-1,1)
        #iv_u_test = OLS_predictions.summary_frame(alpha = alpha).obs_ci_upper.reshape(-1,1)
        
            
    return(OLS_model, OLS_params, mse_train, mse_test, yhat_train, yhat_test)


def SPLS_fitting(X, y, X_test, y_test, K = None, eta = None, eps = 1e-4, maxstep = 1000):
    '''Sparse PLS model fitting based on R libarary

    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    v: int, v fold cross-validation, default = 5
    K: int, the number of latent variable, ranging from 1 to min(m, (v-1)N/v), default v-fold CV, if user-spicified then no CV
    eta: float, sparsity tuning parameter: ranging from 0 to 1, default seq(0,1,0.05) v-fold CV, if user-spicified then no CV

    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: spls model type
    model_params: np_array m x 1
    '''
    
    return SPLS_fitting_method(X, y, X_test, y_test, K = K, eta = eta, maxstep = maxstep)



def EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4):
    '''Elastic Net https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter
    l1_ratio: float, scaling between l1 and l2 penalties, from 0(Ridge) to 1(Lasso)

    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #build model
    EN_model = ElasticNet(random_state = 0, alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter=max_iter, tol = tol)
    EN_model.fit(X, y)
   
    #get paramsters
    EN_params = EN_model.coef_.reshape((-1,1))
   
    #get prediction
    yhat_train = EN_model.predict(X).reshape((-1,1))
    mse_train = mse(y, yhat_train)

    #get prediction for testing
    yhat_test = EN_model.predict(X_test).reshape((-1,1))
    mse_test = mse(y_test, yhat_test)

    return (EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test)


def RR_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4):
    '''Ridge regression
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter
    l1_ratio: float, scaling between l1 and l2 penalties, from 0(Ridge) to 1(Lasso)

    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #build model
    RR_model = Ridge(alpha = alpha, fit_intercept = False).fit(X, y)

    #get paramsters
    RR_params = RR_model.coef_.reshape((-1,1))
       
    #get prediction
    yhat_train = np.dot(X, RR_params).reshape((-1,1))
    mse_train = mse(y, yhat_train)

    #get prediction for testing
    yhat_test = np.dot(X_test, RR_params).reshape((-1,1))
    mse_test = mse(y_test, yhat_test)

    return (RR_model, RR_params, mse_train, mse_test, yhat_train, yhat_test)




def LASSO_fitting(X, y, X_test, y_test, alpha, max_iter = 10000, tol = 1e-4):
    '''Lasso https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter

    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: Lasso model type
    model_params: np_array m x 1
    '''
    
    #build model
    LASSO_model = Lasso(random_state = 0, alpha = alpha, fit_intercept = False, max_iter=max_iter, tol = tol)
    LASSO_model.fit(X, y)
   
    #get paramsters
    LASSO_params = LASSO_model.coef_.reshape((-1,1))
   
    #get prediction
    yhat_train = LASSO_model.predict(X).reshape((-1,1))
    mse_train = mse(y, yhat_train)

    #get prediction for testing
    yhat_test = LASSO_model.predict(X_test).reshape((-1,1))
    mse_test = mse(y_test, yhat_test)

    return (LASSO_model, LASSO_params, mse_train, mse_test, yhat_train, yhat_test)


def ALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, alpha_num = None, cv= False, max_iter = 10000, 
                  tol = 1e-4, selection = 'p_value', select_value = 0.15, trans_type = 'auto'):
    '''Algebric learning via elastic net
    Input:
    X: independent variables of size N x m, has to be non-zscored!
    y: dependent variable of size N x 1, has to be non-zscored!
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter/ int, used when cross-validation, the ith one to use
    l1_ratio: float, scaling between l1 and l2 penalties, from 0(Ridge) to 1(Lasso)
    degree: int, order of nonlinearity you want to consider, can be chosen from 1 - 3
    cv: whether it is cross-validation or final fitting
    selection & select_value:selection ceriteria for the pre-processing step, default: according to 'p-value' with 10% significance
                             'percentage' and the percentatge of variables want to contain
                             'elbow' and use the point with the greatest orthogonal distace from the line linking the first and the last points
                              All the values are calculated based on f-regression (F statistic of univariate linear correlation)
    trans_type: can choose either automatic transformation used in ALVEN ('auto'), or only polynomial transformation ('poly')


                 
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #feature transformation
    if trans_type == 'auto':
        X, X_test = nr.feature_trans(X, X_test, degree = degree, interaction = 'later')
    else:
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    
    
    #remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(X)
    X=sel.transform(X)
    X_test = sel.transform(X_test)
    
    
    #zscore data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    X_test = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)
    
    #eliminate feature
#    if cv:
#        X_e = np.concatenate((X,X_test),axis = 0)
#        y_e = np.concatenate((y,y_test), axis = 0)
#        f_test, p_values = f_regression(X_e, y_e.flatten())
#    else:
#        f_test, p_values = f_regression(X, y.flatten())
  
    #eliminate feature
    f_test, p_values = f_regression(X, y.flatten())
              
        
    if selection == 'p_value':
        X_fit = X[:,p_values<select_value]
        X_test_fit = X_test[:,p_values<select_value]
        retain_index = p_values<select_value
        
    elif selection == 'percentage':
        number = int(math.ceil(select_value * X.shape[1]))
        f_test.sort()
        value = f_test[-number]
        X_fit =  X[:,f_test>=value]
        X_test_fit = X_test[:,f_test>=value]
        
        retain_index = f_test>=value
        
    else:
        f = np.copy(f_test)
        f.sort()  #descending order
        f = f[::-1]
        
        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1),f.reshape(-1,1)),axis=1)
        
        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        
        #find the distance from each point to the line
        vecFromFirst = AllCord- AllCord[0]
        #and calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        
        X_fit =  X[:,f_test>=value]
        X_test_fit = X_test[:,f_test>=value]        
        
        retain_index = f_test>=value
        

    #choose the appropriate alpha in cross_Validation: cv= Ture
    
    if X_fit.shape[1] == 0:
        print('no variable selected by ALVEN')
        ALVEN_model = None
        ALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            X_max = np.concatenate((X_fit,X_test_fit),axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(X_max.T,y_max) ** 2, axis=1)).max())/X_max.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
        
        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(X_fit.T,y) ** 2, axis=1)).max())/X_fit.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
            
        #EN for model fitting
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X_fit, y, X_test_fit, y_test, alpha, l1_ratio, max_iter = max_iter, tol = tol)
        
    
    return (ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index)









def DALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv= False, max_iter = 10000, 
                  tol = 1e-4, selection = 'p_value', select_value = 0.15, trans_type = 'auto'):
    '''Dyanmic Algebric learning via elastic net
    Input:
    X: independent variables of size N x m, has to be non-zscored!
    y: dependent variable of size N x 1, has to be non-zscored!
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter/ int, used when cross-validation, the ith one to use
    l1_ratio: float, scaling between l1 and l2 penalties, from 0(Ridge) to 1(Lasso)
    degree: int, order of nonlinearity you want to consider, can be chosen from 1 - 3
    lag: int, lag of variables you want to consider, xt,xt-1,...xt-l,yt-1,...,yt-l
    cv: whether it is cross-validation or final fitting
    selection & select_value:selection ceriteria for the pre-processing step, default: according to 'p-value' with 10% significance
                             'percentage' and the percentatge of variables want to contain
                             'elbow' and use the point with the greatest orthogonal distace from the line linking the first and the last points
                              All the values are calculated based on f-regression (F statistic of univariate linear correlation)
    trans_type: can choose either automatic transformation used in ALVEN ('auto'), or only polynomial transformation ('poly')


                 
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #feature transformation
    if trans_type == 'auto':
        X, X_test = nr.feature_trans(X, X_test, degree = degree, interaction = 'later')
    else:
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    
    
    
    #lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD,X[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,X_test[lag-1-i:-i-1]))
        
    #lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD,y[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,y_test[lag-1-i:-i-1]))    
    
    #shorterning y
    y = y[lag:]
    y_test = y_test[lag:]
    
    #remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(XD)
    XD=sel.transform(XD)
    XD_test = sel.transform(XD_test)


    #zscore data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)
    

    #eliminate feature
    f_test, p_values = f_regression(XD, y.flatten())
              
        
    if selection == 'p_value':
        XD_fit = XD[:,p_values<select_value]
        XD_test_fit = XD_test[:,p_values<select_value]
        retain_index = p_values<select_value
        
    elif selection == 'percentage':
        number = int(math.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        XD_fit =  XD[:,f_test>=value]
        XD_test_fit = XD_test[:,f_test>=value]
        
        retain_index = f_test>=value
        
    else:
        f = np.copy(f_test)
        f.sort()  #descending order
        f = f[::-1]
        
        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1),f.reshape(-1,1)),axis=1)
        
        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        
        #find the distance from each point to the line
        vecFromFirst = AllCord- AllCord[0]
        #and calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        
        XD_fit =  XD[:,f_test>=value]
        XD_test_fit = XD_test[:,f_test>=value]        
        
        retain_index = f_test>=value
        

    #choose the appropriate alpha in cross_Validation: cv= Ture
    
    if XD_fit.shape[1] == 0:
        print('no variable selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit,XD_test_fit),axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T,y_max) ** 2, axis=1)).max())/XD_max.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
        
        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T,y) ** 2, axis=1)).max())/XD_fit.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
            
        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter = max_iter, tol = tol)
        
        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,(AIC,AICc,BIC))





def DALVEN_testing_kstep(X, y, X_test, y_test, ALVEN_model, retain_index, degree, lag, k_step =1, tol = 1e-4, trans_type = 'auto', plot = False, round_number = ''):
    '''Dyanmic Algebric learning via elastic net for k_step ahead prediction (pre-request: trained DALVEN model)
    Input:
    X: independent variables of size N x m, has to be non-zscored!
    y: dependent variable of size N x 1, has to be non-zscored!
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    ALVEN_model: trained DALVEN model from DALVEN_fitting 
    retain_index: return from DALVEN_fitting in DALVEN_hyper by CV or AIC
    degree: selected degree of nonlinearity in DALVEN_fitting
    lag: selected lag number in DALVEN_fitting
    k_step: positive integer, default =1, number of steps want to predict in to the future
    tol: tolerance for 0-variance feature selection, should be the same as in DALVEN_fitting
    trans_type: transformation type, default = 'auto' is the one include lag, sqrt, 1/x and interactions
    

                 
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #feature transformation
    if trans_type == 'auto':
        X, X_test = nr.feature_trans(X, X_test, degree = degree, interaction = 'later')
    else:
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
     
    
    #lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD,X[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,X_test[lag-1-i:-i-1]))
        
    #lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD,y[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,y_test[lag-1-i:-i-1]))    
    
    #shorterning y
    y = y[lag:]
    y_test = y_test[lag:]
    
    #remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(XD)
    XD=sel.transform(XD)
    XD_test = sel.transform(XD_test)

    position = XD.shape[1]-lag
    #zscore data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)
    

    #eliminate feature      
    XD_test_fit = XD_test[:,retain_index]        
        
    #0-step results
    yhat_test_multi = {}
    mse_test_multi = np.zeros((k_step,1))

    yhat_test_multi[0] = ALVEN_model.predict(XD_test_fit).reshape((-1,1))
    mse_test_multi[0] = mse(y_test, yhat_test_multi[0])
    

    
    k_step = k_step -1
    #multi-step prediction
    for k in range(k_step):
        #lag padding for y in design matrix
        XD_test = XD_test[1:]
        for l in range(min(lag,k+1)):
            XD_test[:,position+l] =  yhat_test_multi[k-l][:-1-l].flatten()
        XD_test_fit = XD_test[:,retain_index]        

        yhat_test_multi[k+1] = ALVEN_model.predict(XD_test_fit).reshape((-1,1))
        mse_test_multi[k+1] = mse(y_test[k+1:], yhat_test_multi[k+1])
        
        
        
    ##plot results
    if plot:
        if X.shape[0] == X_test.shape[0]:
            if abs(np.sum(X-X_test))<tol:
                my_data = 'train'
            else:
                my_data = 'test'
        else:
            my_data = 'test'
            
        print('=============Plot Results==============')
        import matplotlib.pyplot as plt
        s=12
        plt.figure(figsize=(3,2))
        plt.plot(mse_test_multi, 'd-')
        plt.title('MSE for y ' + my_data + ' prediction', fontsize = s)
        plt.xlabel('k-step ahead', fontsize = s)
        plt.ylabel('MSE', fontsize = s)
        plt.savefig('MSE_'+my_data+round_number+'_DALVEN.png', dpi=600,bbox_inches='tight')
        
        
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
        
        #plot the prediction vs real
        for i in range(k_step+1):
            plt.figure(figsize=(5,3))
            plt.plot(y_test[i+1:], color= cmap(1), label= 'real')
            plt.plot(yhat_test_multi[i][1:], '--',color= 'xkcd:coral', label = 'prediction')
            plt.title(my_data + ' data ' + str(i+1) +'-step prediction',fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y',fontsize=s)
            plt.legend(fontsize=s)
            plt.tight_layout()                    
            plt.savefig('DALVEN_'+my_data+'_step_'+str(i+1)+ round_number+'.png', dpi = 600,bbox_inches='tight')

                
        
        
    
    return (mse_test_multi, yhat_test_multi)









##########################################################################################
def DALVEN_fitting_full_nonlinear(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv= False, max_iter = 10000, 
                                  tol = 1e-4, selection = 'p_value', select_value = 0.05, trans_type = 'auto'):
    '''Dyanmic Algebric learning via elastic net with fully nonlienar mapping fo both x and y and interactions
    Input:
    X: independent variables of size N x m, has to be non-zscored!
    y: dependent variable of size N x 1, has to be non-zscored!
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    alpha: float, regularization parameter/ int, used when cross-validation, the ith one to use
    l1_ratio: float, scaling between l1 and l2 penalties, from 0(Ridge) to 1(Lasso)
    degree: int, order of nonlinearity you want to consider, can be chosen from 1 - 3
    lag: int, lag of variables you want to consider, xt,xt-1,...xt-l,yt-1,...,yt-l
    cv: whether it is cross-validation or final fitting
    selection & select_value:selection ceriteria for the pre-processing step, default: according to 'p-value' with 10% significance
                             'percentage' and the percentatge of variables want to contain
                             'elbow' and use the point with the greatest orthogonal distace from the line linking the first and the last points
                              All the values are calculated based on f-regression (F statistic of univariate linear correlation)
    trans_type: can choose either automatic transformation used in ALVEN ('auto'), or only polynomial transformation ('poly')


                 
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    #lag design matrix first
    #lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD,X[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,X_test[lag-1-i:-i-1]))
        
    #lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD,y[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,y_test[lag-1-i:-i-1]))    
    
    
    
    #nonliner mapping
        #feature transformation
    if trans_type == 'auto':
        XD, XD_test = nr.feature_trans(XD, XD_test, degree = degree, interaction = 'later')
    else:
        XD, XD_test = nr.poly_feature(XD, XD_test, degree = degree, interaction = True, power = True)
    
  
    #remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(XD)
    XD=sel.transform(XD)
    XD_test = sel.transform(XD_test)

   
    #shorterning y
    y = y[lag:]
    y_test = y_test[lag:]
    
    
    #zscore data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)
    

    #eliminate feature
    f_test, p_values = f_regression(XD, y.flatten())
              
        
    if selection == 'p_value':
        XD_fit = XD[:,p_values<select_value]
        XD_test_fit = XD_test[:,p_values<select_value]
        retain_index = p_values<select_value
        
    elif selection == 'percentage':
        number = int(math.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        XD_fit =  XD[:,f_test>=value]
        XD_test_fit = XD_test[:,f_test>=value]
        
        retain_index = f_test>=value
        
    else:
        f = np.copy(f_test)
        f.sort()  #descending order
        f = f[::-1]
        
        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1),f.reshape(-1,1)),axis=1)
        
        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        
        #find the distance from each point to the line
        vecFromFirst = AllCord- AllCord[0]
        #and calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        
        XD_fit =  XD[:,f_test>=value]
        XD_test_fit = XD_test[:,f_test>=value]        
        
        retain_index = f_test>=value
        

    #choose the appropriate alpha in cross_Validation: cv= Ture
    
    if XD_fit.shape[1] == 0:
        print('no variable selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit,XD_test_fit),axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T,y_max) ** 2, axis=1)).max())/XD_max.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
        
        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T,y) ** 2, axis=1)).max())/XD_fit.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]
            
        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter = max_iter, tol = tol)
        
        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,(AIC,AICc,BIC))









def DALVEN_testing_kstep_full_nonlinear(X, y, X_test, y_test, ALVEN_model, retain_index, degree, lag, k_step =1, tol = 1e-4, trans_type = 'auto', plot = False, round_number = ''):
    '''Dyanmic Algebric learning via elastic net for k_step ahead prediction (pre-request: trained DALVEN model with full nonlinearity)
    Input:
    X: independent variables of size N x m, has to be non-zscored!
    y: dependent variable of size N x 1, has to be non-zscored!
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    ALVEN_model: trained DALVEN model from DALVEN_fitting 
    retain_index: return from DALVEN_fitting in DALVEN_hyper by CV or AIC
    degree: selected degree of nonlinearity in DALVEN_fitting
    lag: selected lag number in DALVEN_fitting
    k_step: positive integer, default =1, number of steps want to predict in to the future
    tol: tolerance for 0-variance feature selection, should be the same as in DALVEN_fitting
    trans_type: transformation type, default = 'auto' is the one include lag, sqrt, 1/x and interactions
    

                 
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: EN model type
    model_params: np_array m x 1
    '''
    
    #lag design matrix first
    #lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD,X[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,X_test[lag-1-i:-i-1]))
        
    #lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD,y[lag-1-i:-i-1]))
        XD_test = np.hstack((XD_test,y_test[lag-1-i:-i-1]))    
    
    
    
    #nonliner mapping
        #feature transformation
    if trans_type == 'auto':
        XD, XD_test = nr.feature_trans(XD, XD_test, degree = degree, interaction = 'later')
    else:
        XD, XD_test = nr.poly_feature(XD, XD_test, degree = degree, interaction = True, power = True)
    
  
    #remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(XD)
    XD=sel.transform(XD)
    XD_test = sel.transform(XD_test)

   
    #shorterning y
    y = y[lag:]
    y_test_ori = y_test[:]
    y_test = y_test[lag:]
    


    #zscore data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y_test = scaler_y.transform(y_test)
#    y_test_ori = scaler_y.transform(y_test_ori)

    #eliminate feature      
    XD_test_fit = XD_test[:,retain_index]        
        
    #0-step results
    yhat_test_multi = {}
    mse_test_multi = np.zeros((k_step,1))

    yhat_test_multi[0] = ALVEN_model.predict(XD_test_fit).reshape((-1,1))
    mse_test_multi[0] = mse(y_test, yhat_test_multi[0])
    

#    print('starting k step prediction')
    k_step = k_step -1
    #multi-step prediction######################
    for k in range(k_step):
#        print(k+2)
        #################mapping
        XD_test = X_test[lag+k+1:]
        for i in range(lag):
            XD_test = np.hstack((XD_test,X_test[lag+k-i:-i-1]))
        
        position = XD_test.shape[1]
        #lag padding for y in design matrix
        for i in range(lag):
            XD_test = np.hstack((XD_test,y_test_ori[lag+k-i:-i-1]))    
        
        for l in range(min(lag,k+1)):
            y_feed= yhat_test_multi[k-l]
            y_feed=scaler_y.inverse_transform(y_feed)
            XD_test[:,position+l] =  y_feed[:-1-l].flatten()        
        
        #nonliner mapping
            #feature transformation
        if trans_type == 'auto':
            XD_test,_ = nr.feature_trans(XD_test, degree = degree, interaction = 'later')
        else:
            XD_test ,_= nr.poly_feature(XD_test,  degree = degree, interaction = True, power = True)
        
      
        #remove feature with 0 variance
        XD_test = sel.transform(XD_test)
     
        XD_test = scaler_x.transform(XD_test)


        XD_test_fit = XD_test[:,retain_index]        

  

        yhat_test_multi[k+1] = ALVEN_model.predict(XD_test_fit).reshape((-1,1))
        mse_test_multi[k+1] = mse(y_test[k+1:], yhat_test_multi[k+1])
        
        
        
    ##plot results
    if plot:
        if X.shape[0] == X_test.shape[0]:
            if abs(np.sum(X-X_test))<tol:
                my_data = 'train'
            else:
                my_data = 'test'
        else:
            my_data = 'test'
            
        print('=============Plot Results==============')
        import matplotlib.pyplot as plt
        s=12
        plt.figure(figsize=(3,2))
        plt.plot(mse_test_multi, 'd-')
        plt.title('MSE for y ' + my_data + ' prediction', fontsize = s)
        plt.xlabel('k-step ahead', fontsize = s)
        plt.ylabel('MSE', fontsize = s)
        plt.savefig('MSE_'+my_data+ round_number+'_DALVEN.png', dpi=600,bbox_inches='tight')
        
        
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
        
        #plot the prediction vs real
        for i in range(k_step+1):
            plt.figure(figsize=(5,3))
            plt.plot(y_test[i+1:], color= cmap(1), label= 'real')
            plt.plot(yhat_test_multi[i][1:], '--',color= 'xkcd:coral', label = 'prediction')
            plt.title(my_data + ' data ' + str(i+1) +'-step prediction',fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y',fontsize=s)
            plt.legend(fontsize=s)
            plt.tight_layout()                    
            plt.savefig('DALVEN_'+my_data+'_step_'+str(i+1)+ round_number+'.png', dpi = 600,bbox_inches='tight')

                
        
        
    
    return (mse_test_multi, yhat_test_multi)


    