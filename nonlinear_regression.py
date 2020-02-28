# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:00:27 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

'''This file is for nonlinear regression with symbolic transformation
   combined with LASSO, SPLS, EN'''

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
#from copy import deepcopy


def _xexp(x):
    '''exponential transform with protection against large numbers'''
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 9, np.exp(x), np.exp(9)*np.ones_like(x))

def _xlog(x):
    '''logarithm with protection agiasnt small numbers'''
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return np.where(np.abs(x) > np.exp(-10), np.log(abs(x)), -10*np.ones_like(x))
 
def _xsqrt(x):
    '''square root with protection with negative values (take their abs)'''
    with np.errstate(invalid = 'ignore'):
        return np.sqrt(np.abs(x))
    
def _xinv(x):
    '''inverse with protection with 0 value'''
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return np.where(np.abs(x)>1e-9, 1/x, 1e9*np.ones_like(x))


def _mul(X,Y):
    '''column-wise multiplication'''
    

def poly_feature(X, X_test = None, degree = 2, interaction = True, power = True):
    '''
    This function transforms features to polynomials according to the specified degree
    
    Input:
    X: N x m np_array indepdendent variables
    X_test: independent variables of size N_test x m np_array
    degree: int, degree of the polynomials, default 2
    interaction: Bool, including interactions (x1, x2) to (x1, x2, x1x2)
    power: Bool, including powers (x1 x2) to (x1**2, x2**2)
    
    Return:
    transofrmed X, np_array, N x m_trans
    transformed X_test, np_array, N_test x m_trans
    '''
    if interaction and power:
        poly = PolynomialFeatures(degree,include_bias=False)
        X = poly.fit_transform(X)
        if X_test is not None:
            X_test = poly.fit_transform(X_test)
        
    if interaction and not power:
        poly = PolynomialFeatures(degree,include_bias=False, interaction_only = True)
        X = poly.fit_transform(X)
        if X_test is not None:
            X_test = poly.fit_transform(X_test)
        
    if not interaction and power:
        X_copy = X[:]
        if X_test is not None:
            X_test_copy = X_test[:]
        for i in range(1,degree):
            X = np.column_stack((X, X_copy**(i+1)))
            if X_test is not None:
                X_test = np.column_stack((X_test, X_test_copy**(i+1)))
    
    return (X, X_test)

        
def feature_trans(X, X_test = None, degree = 2, interaction = 'later'):
    '''
    This function transforms features according to the specified nonlinear forms
    including [poly, log, exp, sigmoid, abs, 1/x, sqrt(x)]
    
    Input:
    X: N x m np_array indepdendent variables
    X_test: independent variables of size N_test x m np_array
    
    form: list including str of the nonlinear forms
    
    '''
    
    Xlog = _xlog(X)
    Xinv = _xinv(X)    
    Xsqrt = _xsqrt(X)
    
    
#    Xexp = _xexp(X)
#    Xexp_t = _xexp(X_test)
        
#    Xsig = 1/(1+ _xexp(-X))
#    Xsig_t = 1/(1+ _xexp(-X_test))
    
#    Xabs = np.abs(X)
#    Xabs_t = np.abs(X_test)
    

    
    if X_test is not None:
        Xlog_t = _xlog(X_test)
        Xsqrt_t = _xsqrt(X_test)
        Xinv_t = _xinv(X_test)
    
      
    
    if interaction == 'no':
        if degree == 1:
            X = np.column_stack((X, Xlog, Xsqrt, Xinv))
            if X_test is not None:
                X_test = np.column_stack((X_test, Xlog_t, Xsqrt_t, Xinv_t))
        
        if degree == 2:
            X = np.column_stack((X, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv ))
            
            
            if X_test is not None:
                X_test = np.column_stack((X_test, Xlog_t, Xsqrt_t, Xinv_t, X_test**2,Xlog_t**2,Xinv_t**2, X_test*Xsqrt_t, Xlog_t*Xinv_t, Xsqrt_t*Xinv_t ))
        
        if degree == 3:
            X = np.column_stack((X, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv,
                                 X**3, Xlog**3, Xinv**3, X**2*Xsqrt, Xlog**2*Xinv, Xlog*Xsqrt*Xinv,Xlog*Xinv**2, Xsqrt*Xinv**2))
            
            if X_test is not None:
                X_test = np.column_stack((X_test, Xlog_t, Xsqrt_t, Xinv_t, X_test**2,Xlog_t**2,Xinv_t**2, X_test*Xsqrt_t, Xlog_t*Xinv_t, Xsqrt_t*Xinv_t,
                                     X_test**3, Xlog_t**3, Xinv_t**3, X_test**2*Xsqrt_t, Xlog_t**2*Xinv_t, Xlog_t*Xsqrt_t*Xinv_t,
                                     Xlog_t*Xinv_t**2, Xsqrt_t*Xinv_t**2))
    
                
    if interaction == 'later':
        if degree == 1:
            X = np.column_stack((X, Xlog, Xsqrt, Xinv))
            if X_test is not None:
                X_test = np.column_stack((X_test, Xlog_t, Xsqrt_t, Xinv_t))
            
        
        if degree == 2:
            poly = PolynomialFeatures(degree = 2,include_bias=False, interaction_only = True)
            X_inter = poly.fit_transform(X)[:,X.shape[1]:]

            X = np.column_stack((X, X_inter, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv))
            
            if X_test is not None:
                X_test_inter = poly.fit_transform(X_test)[:,X_test.shape[1]:]
                     
                X_test = np.column_stack((X_test, X_test_inter, Xlog_t, Xsqrt_t, Xinv_t, X_test**2,Xlog_t**2,Xinv_t**2, X_test*Xsqrt_t, Xlog_t*Xinv_t, Xsqrt_t*Xinv_t ))
    
        if degree == 3:
            poly = PolynomialFeatures(degree = 3,include_bias=False, interaction_only = True)
            X_inter = poly.fit_transform(X)[:,X.shape[1]:]
            
            X = np.column_stack((X,X_inter, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv,
                                 X**3, Xlog**3, Xinv**3, X**2*Xsqrt, Xlog**2*Xinv, Xlog*Xsqrt*Xinv,Xlog*Xinv**2, Xsqrt*Xinv**2))

            if X_test is not None:
                 X_test_inter = poly.fit_transform(X_test)[:,X_test.shape[1]:]

                 X_test = np.column_stack((X_test, X_test_inter, Xlog_t, Xsqrt_t, Xinv_t, X_test**2,Xlog_t**2,Xinv_t**2, X_test*Xsqrt_t,  Xlog_t*Xinv_t, Xsqrt_t*Xinv_t,
                                     X_test**3, Xlog_t**3, Xinv_t**3, X_test**2*Xsqrt_t, Xlog_t**2*Xinv_t, Xlog_t*Xsqrt_t*Xinv_t,
                                     Xlog_t*Xinv_t**2, Xsqrt_t*Xinv_t**2))
    
    return (X, X_test)


