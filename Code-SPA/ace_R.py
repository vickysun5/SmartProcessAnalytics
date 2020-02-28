# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:11:58 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

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
    ace = importr('acepack', robject_translations = d, lib_loc = "/Users/Vicky/Anaconda3/Lib/R/library")
except:
    ace = importr('acepack', robject_translations = d, lib_loc = "/Users/Vicky/Anaconda3/envs/rstudio/lib/R/library")


def ace_R(x, y, cat = None):
    '''
    x: one predictor, Nx1
    y: one response, Nx1
    categorical:wheter variables are categorical, [y, x], integer vector
    
    '''
    
    '''Data preparation'''
    rpy2.robjects.numpy2ri.activate()
    
    #convert training data numpy to R vector/matrix
    nrx,ncx = x.shape
    xr = ro.r.matrix(x,nrow=nrx,ncol=ncx)
    ro.r.assign("x", xr)
    
    nry,ncy = y.shape
    yr = ro.r.matrix(y,nrow=nry,ncol=ncy)
    ro.r.assign("y", yr)
    
    #calculate transofrmation
    if cat is not None:
        a = ace.ace(xr,yr, cat = 1)
    else:
        a = ace.ace(xr,yr)
        
    tx = a.rx2('tx')
    ty = a.rx2('ty')
    
    #calculate final correlation coefficient
    corr = np.array(ro.r.cor(tx,ty))[0][0]
    
    
    return corr
