# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:33:24 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

'''This file creates complex nonlinear regression models, which are optional to
the final software'''

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def model_getter(model_name):
    '''Return the model according to the name'''
    
    switcher = {
            'RF': RF_fitting,
            'SVR': SVR_fitting
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



def RF_fitting(X, y, X_test, y_test, n_estimators = 100, max_depth = 10, min_samples_leaf = 0.1, max_features = 'auto',random_state=0):
    '''Random forest regressor https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.decision_path
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    n_estimators: int, number of trees in the RF
    max_depth: int, max_depth of a single tree
    max_features: maximum number of features when considered for a potential splitting, 'auto' = m
    random_state: int, if None, np.rand is used
        
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: RF model type
    '''
    
    #build model
    RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,random_state= random_state,
                                    max_features = max_features, min_samples_leaf = min_samples_leaf)
    RF.fit(X, y.flatten())
    
    #predict
    yhat_train = RF.predict(X).reshape((-1,1))
    yhat_test = RF.predict(X_test).reshape((-1,1))
    

    #get error
    mse_train = mse(y, yhat_train)
    mse_test = mse(y_test, yhat_test)

    return (RF, mse_train, mse_test, yhat_train, yhat_test)




def MLP_fitting(X, y, X_test, y_test, hidden_layer_sizes = (10,), activation = 'tanh', solver = 'adma', alpha = 0.0001,
        learning_rate_init = 0.001, max_iter = 1000, random_state = 0, tol = 1e-4, early_stopping = True, n_iter_no_change = 5, validation_fraction = 0.1):
    '''A MLP with adam solver https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    hidden_layer_sizes: tuple, number of nodes in each hindden layers
    activation： activation function
    alpha: float, L2 regularization parameter
    Other parameters see the link above
        
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: MLP model type
    '''
    
    #build model
    MLP = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver, alpha = alpha,learning_rate_init=learning_rate_init,
                       max_iter = max_iter, random_state = random_state, tol = tol, early_stopping = early_stopping, n_iter_no_change = n_iter_no_change, validation_fraction =validation_fraction)
    MLP.fit(X,y)
    
    #predict
    yhat_train = MLP.predict(X).reshape(-1,1)
    yhat_test = MLP.predict(X_test).reshape(-1,1)
    

    #get error
    mse_train = mse(y, yhat_train)
    mse_test = mse(y_test, yhat_test)
   
    return (MLP,  mse_train, mse_test, yhat_train, yhat_test)



def SVR_fitting(X, y, X_test, y_test, C = 100, epsilon = 10, gamma = 'auto', tol = 1e-4, max_iter = 10000):
    '''Support Vector Reression with RBF kernel https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    Input:
    X: independent variables of size N x m
    y: dependent variable of size N x 1
    X_test: independent variables of size N_test x m
    y_test: dependent variable of size N_test x 1
    C： float, penalty parameter of the error term
    epsilon: float, epsilon-tube within which no penalty is associated
    gamma: float, kernal coefficient for 'rbf'
        
    Output:
    tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
    trained_model: SVR model type
    '''
    
    #build model
    SVR_model = SVR(gamma=gamma, C=C, epsilon=epsilon, tol = tol, max_iter = max_iter)
    SVR_model.fit(X, y.flatten())
    
    #predict
    yhat_train = SVR_model.predict(X).reshape((-1,1))
    yhat_test = SVR_model.predict(X_test).reshape((-1,1))
    

    #get error
    mse_train = mse(y, yhat_train)
    mse_test = mse(y_test, yhat_test)

    return (SVR_model, mse_train, mse_test, yhat_train, yhat_test)

