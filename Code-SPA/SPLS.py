# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:58:29 2018

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""
#check the lib
#import rpy2.rinterface
#rpy2.rinterface.set_initoptions((b'rpy2', b'--no-save', b'--no-restore', b'--quiet'))
#from rpy2.robjects.packages import importr
#base = importr('base')
#print(base._libPaths())

#check in R
#.libPaths()
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri


'''Load spls package from R libarary
   need to be installed at the right location'''
utils = importr("utils")
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}

try:
    '''default R libarary path'''
    spls = importr('spls', robject_translations = d, lib_loc = "/Users/Vicky/Anaconda3/Lib/R/library")
except:
    spls = importr('spls', robject_translations = d, lib_loc = "/Users/Vicky/Anaconda3/envs/rstudio/lib/R/library")


def mse(y, yhat):
    """
    This function calculate the goodness of fit mse
    Input: y: N x 1 real response
           yhat: N x 1 predited by the model
           
    Output: mse float
    """
    return np.sum((yhat-y)**2)/y.shape[0]


def SPLS_fitting_method(X, y, X_test, y_test, K = None, eta = None, v = 5, eps = 1e-4, maxstep = 1000):
    """
    This function use rpy2 interface to call R package "spls"
    https://cran.r-project.org/web/packages/spls/vignettes/spls-example.pdf
    "The responses and the predictors are assumed to be numerical and should not contain missing values.
    As part of pre-processing, the predictors are centered and scaled and the responses are centered
    automatically as default by the package ‘spls’"

    Input:
        X: independent variables of size N x m
        y: dependent variable of size N x 1
        X_test: independent variables of size N_test x m
        y_test: dependent variable of size N_test x 1
        v: int, v fold cross-validation, default = 5 using the default CV in SPLS package
        K: int, the number of latent variable, ranging from 1 to min(m, (v-1)N/v), default using cross validation to determin
        eta: float, sparsity tuning parameter: ranging from 0 to 1, default seq(0,1,0.05), default, using cross validation to detemine
        
    Output:
        tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
        trained_model: spls model type
        model_params: np_array m x 1
        """

    '''Data preparation'''
    rpy2.robjects.numpy2ri.activate()
    
    #convert training data numpy to R vector/matrix
    nr,nc = X.shape
    Xr = ro.r.matrix(X, nrow=nr, ncol=nc)
    ro.r.assign("X", Xr)
    
#    nr_test,nc_test = X_test.shape
#    Xr_test = ro.r.matrix(X_test, nrow=nr_test, ncol=nc_test)
#    ro.r.assign("X_test", Xr_test)

    nry,ncy = y.shape
    yr = ro.r.matrix(y,nrow=nry,ncol=ncy)
    ro.r.assign("y", yr)
    
#    nry_test,ncy_test = y_test.shape
#    yr_test = ro.r.matrix(y_test,nrow=nry_test,ncol=ncy_test)
#    ro.r.assign("y_test", yr_test)
    
    '''CV fitting to choose K and eta if not given'''
    if K == None and eta == None:
        m = nc
        N = nr
        f = spls.cv_spls(Xr, yr, K=ro.r.seq(1,min(m,int((v-1)/v*N)),1), eta=ro.r.seq(0, 0.95, 0.05), fold=5, plot_it = False, scale_x=False, scale_y=False, eps=eps)
        #mse_map = np.array(f.rx2('mspemat'))  #access class trhough .rx2, converting back to numpy
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    elif K== None and eta != None:
        m = nc
        N = nr
        f = spls.cv_spls(Xr, yr, K=ro.r.seq(1,min(m,int((v-1)/v*N)),1), eta=eta, fold=5, plot_it = False, scale_x=False, scale_y=False, eps=eps)
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    elif K != None and eta == None:
        f = spls.cv_spls(Xr, yr, K=K, eta=ro.r.seq(0, 1, 0.05), fold=5, plot_it = False, scale_x=False, scale_y=False)
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    else:
        '''If specified, not using the default cross-validation in SPLS package'''
        K_opt = K
        eta_opt = eta
    
    '''Fit the final model'''
    SPLS_model = spls.spls(Xr, yr, eta = eta_opt, K = K_opt, scale_x=False, scale_y=False, eps=eps, maxstep = maxstep)

    '''Extract coefficients'''
    SPLS_params = spls.predict_spls(SPLS_model, type = "coefficient")
    SPLS_params = np.array(SPLS_params)

    '''Make predictions of training data'''
#    yhat_train = spls.predict_spls(SPLS_model, type = "fit")
#    yhat_train = np.array(yhat_train)
    yhat_train = np.dot(X, SPLS_params)
   
    '''Make prediction of testing data'''
#    yhat_test = spls.predict_spls(SPLS_model, Xr_test, type = "fit")
#    yhat_test = np.array(yhat_test)
    yhat_test = np.dot(X_test, SPLS_params)


    '''Calculate mse'''
    mse_train = mse(y, yhat_train)
    mse_test = mse(y_test, yhat_test)


    return (SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test)


