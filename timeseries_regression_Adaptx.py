# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:38:56 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

"""
This file call the matlab based ADPATx for state space model fitting,
There are two modes:
    (1) Single training set, with option of testing data
    (2) Multiple training sets, with option of testing data
"""

import numpy as np
import matlab.engine
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def Adaptx_matlab_single(X, y, data_url, url, X_test=None, y_test=None, train_ratio = 1,\
                      mymaxlag = 12, mydegs = [-1, 0, 1], mynow = 1, steps = 10, plot = True):
    '''This function fits the CVA-state space model for training data X, y of first training ratio data,
    and use rest (1-ratio_data) for forcasting. There can also be test data to test the fitted state space model
    Input:
        X: training data predictors numpy array: Nxm
        y: training data response numy array: Nx1
        X_test: testing data predictors numpy arrray: N_testxm
        y_test: testing data response numpy array: N_test x 1
        data_url: desired working directory for saving all the results, be a sub-folder of the main ADPATX folder
        url: main directory for ADAPTX folder
        train_ratio: float, portion of training data used to train the model, and the rest is used as validation data
        mymaxlag: maximum lag number considered for the CVA, default = 12
        mydegs: degs considered for the trend in state space model, can chose from [-1 0 1 2 3,4], defualt [-1 0 1]
        mynow: instantaneous effect of u on y in state space model, 1: yes, 0: no, defualt 1
        steps: number of steps considered for prediction
        plot: flag for plotting results or not, default TRUE
        
        
    Output:
        optimal_params: dictonary
            mylag: int, selected optimal lag number by AIC
            mydeg: int, selected optimal order on trend by AIC
            ord: int, selected optimal order for CVA by AIC
        
        myresults:
            State space model parameters:
                Phi, G, H, A, B, Q, R, E, F, ABR, BBR
            CVA projection: Jk
            
        Preditction results:
            MSE_train, MSE_val, MSE_test with differnt prediction steps
            
                Ym: prediction by final optimal model, Num_steps X timeinstances, the frist row is one step ahead by Kalman
                error: the error Ym-Yp

    '''

    ###Save Data in desired form to mydata.mat, which contains mydata as a matrix of size (m_y+m_x)xN, where y as the first row
    n_train = round(train_ratio*np.shape(y)[0])
    
    scaler = StandardScaler()
    scaler.fit(X[:n_train])
    X = scaler.transform(X)
        
    scalery = StandardScaler()
    scalery.fit(y[:n_train])
    y=scalery.transform(y)
        
    
    
    mydata=np.vstack((np.transpose(y),np.transpose(X)))    
    sio.savemat('mydata.mat', {'mydata':mydata})
    
    if y_test is not None:
        ###Save test Data in desired form to mydataval.mat
        X_test = scaler.transform(X_test)
        y_test = scalery.transform(y_test)
        
        
        mydataval=np.vstack((np.transpose(y_test),np.transpose(X_test)))    
        sio.savemat('mydataval.mat', {'mydataval':mydataval})
        test = 1
    else:
        test = 0
    
    
    m_y = np.shape(y)[1]
    ###Save parameters in a file
    sio.savemat('myparams.mat', {'url':url,'data_url':data_url, 'mydimy':m_y, 'mymaxlag':mymaxlag,'mydegs':mydegs, 'mynow':mynow, 'val':test, 'steps':steps, 'n_train':n_train})
     
    
    ###Call the matlab script
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())
    #eng.addpath(url, nargout=0)
    eng.CVA_singleset_fit_test(nargout=0)
    
    eng.quit()
    
    ###Read Results and do Plots
    #the parameters are saved in myresults
    myresults = sio.loadmat(data_url+'myresults.mat')
    
    
    prediction_train = sio.loadmat(data_url+'kstep_training.mat')
        
    y_real_train = np.array(prediction_train['yp'])
    if train_ratio < 1:
        y_real_val = y_real_train[:,n_train:]
        y_real_train = y_real_train[:,:n_train]
    
    y_predict_train = np.array(prediction_train['ym'])
    if train_ratio < 1:
        y_predict_val = y_predict_train[:,n_train:]
        y_predict_train = y_predict_train[:,:n_train]
    else:
        y_predict_val = None
    
    train_error = np.array(prediction_train['ye'])
    if train_ratio < 1:
        val_error = train_error[:,n_train:]
        train_error = train_error[:,:n_train]
        MSE_val = np.nansum(val_error**2,axis=1)/np.sum(~np.isnan(val_error),axis=1)
    else:
        MSE_val = None
        val_error = None
        
    MSE_train = np.nansum(train_error**2,axis=1)/np.sum(~np.isnan(train_error),axis=1)
        
    if test:
        prediction_test = sio.loadmat(data_url+'kstep_testing.mat')
        y_real_test = np.array(prediction_test['yp'])
        y_predict_test = np.array(prediction_test['ym'])
        test_error = np.array(prediction_test['ye'])    
        MSE_test = np.nansum(test_error**2,axis=1)/np.sum(~np.isnan(test_error),axis=1)
    else:
        MSE_test = None
        test_error = None
        y_predict_test = None
        
    
    
    #plot the prediction results
    if plot:
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
        
        s=12
        
        #plot the prediction vs real
        for i in range(steps):
            for j in range(m_y):
        
                plt.figure(figsize=(5,3))
                plt.plot(y_real_train[j], color= cmap(j*2+1), label= 'real')
                plt.plot(y_predict_train[m_y*i+j], '--', color= 'xkcd:coral', label = 'prediction')
                plt.title('Training data' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()
                plt.savefig('Train_var_' + str(j+1)+'_step_'+str(i+1)+'.png', dpi = 600,bbox_inches='tight')
                
                if train_ratio < 1:
                    plt.figure(figsize=(5,3))
                    plt.plot(y_real_val[j], color= cmap(j*2+1), label= 'real')
                    plt.plot(y_predict_val[m_y*i+j], '--', color= 'xkcd:coral',label = 'prediction')
                    plt.title('Validation data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                    plt.xlabel('Time index',fontsize=s)
                    plt.ylabel('y',fontsize=s)
                    plt.legend(fontsize=s)
                    plt.tight_layout()                    
                    plt.savefig('Val_var_' + str(j+1)+'_step_'+str(i+1)+'.png', dpi = 600,bbox_inches='tight')

                if test:
                    plt.figure(figsize=(5,3))
                    plt.plot(y_real_test[j], color= cmap(j*2+1), label= 'real')
                    plt.plot(y_predict_test[m_y*i+j], '--',color= 'xkcd:coral', label = 'prediction')
                    plt.title('Test data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                    plt.xlabel('Time index',fontsize=s)
                    plt.ylabel('y',fontsize=s)
                    plt.legend(fontsize=s)
                    plt.tight_layout()                    
                    plt.savefig('Test_var_' + str(j+1)+'_step_'+str(i+1)+'.png', dpi = 600,bbox_inches='tight')

                
#                plt.close('all')
        
        #plot fitting errors
        
        max_limit=np.nanmax(train_error[-2:],axis=1)
        min_limit=np.nanmin(train_error[-2:],axis=1)
        
        fig, axs = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
        
        if m_y>1:
            for i in range(steps):
                for j in range(m_y):
                    axs[i,j].plot(train_error[m_y*i+j], color= cmap(j*2+1))
                    axs[i,j].set_title('Training data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                    if i is steps-1:
                        axs[i,j].set_xlabel('Time index', fontsize=s)              
            fig.tight_layout()
            plt.savefig('Train error.png', dpi = 600,bbox_inches='tight')
            
            
            if train_ratio < 1: 
                
                max_limit=np.nanmax(val_error[-2:],axis=1)
                min_limit=np.nanmin(val_error[-2:],axis=1)
                fig1, axs1 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
                
                for i in range(steps):
                    for j in range(m_y):
                        axs1[i,j].plot(val_error[m_y*i+j], color= cmap(j*2+1))
                        axs1[i,j].set_title('Val data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                        axs1[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                        if i is steps-1:
                            axs1[i,j].set_xlabel('Time index', fontsize=s)                
                fig1.tight_layout()
                plt.savefig('Val error.png', dpi=600,bbox_inches='tight')
                
                
            if test:  
                max_limit=np.nanmax(test_error[-2:],axis=1)
                min_limit=np.nanmin(test_error[-2:],axis=1)
                fig2, axs2 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
                
                for i in range(steps):
                    for j in range(m_y):
                        axs2[i,j].plot(test_error[m_y*i+j], color= cmap(j*2+1))
                        axs2[i,j].set_title('Test data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                        axs2[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                        if i is steps-1:
                            axs2[i,j].set_xlabel('Time index', fontsize=s)                
                fig2.tight_layout()
                plt.savefig('Test error.png', dpi=600,bbox_inches='tight')        
        else:
            j=0
            for i in range(steps):
                axs[i].plot(train_error[m_y*i+j], color= cmap(j*2+1))
                axs[i].set_title('Training data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                axs[i].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                if i is steps-1:
                    axs[i].set_xlabel('Time index', fontsize=s)              
            fig.tight_layout()
            plt.savefig('Train error.png', dpi = 600,bbox_inches='tight')
            
            
            if train_ratio < 1:               
                max_limit=np.nanmax(val_error[-2:],axis=1)
                min_limit=np.nanmin(val_error[-2:],axis=1)
                fig1, axs1 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
                
                for i in range(steps):                   
                    axs1[i].plot(val_error[m_y*i+j], color= cmap(j*2+1))
                    axs1[i].set_title('Val data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs1[i].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                    if i is steps-1:
                        axs1[i].set_xlabel('Time index', fontsize=s)                
                fig1.tight_layout()
                plt.savefig('Val error.png', dpi=600,bbox_inches='tight')
                
                
            if test:  
                max_limit=np.nanmax(test_error[-2:],axis=1)
                min_limit=np.nanmin(test_error[-2:],axis=1)
                fig2, axs2 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
                
                for i in range(steps):
                    axs2[i].plot(test_error[m_y*i+j], color= cmap(j*2+1))
                    axs2[i].set_title('Test data' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs2[i].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                    if i is steps-1:
                        axs2[i].set_xlabel('Time index', fontsize=s)                
                fig2.tight_layout()
                plt.savefig('Test error.png', dpi=600,bbox_inches='tight')        
        
        #MSE for prediction results over different steps
        for i in range(m_y):
            plt.figure(figsize=(3,2))
            plt.plot(MSE_train[i::m_y], 'd-', color = cmap(i*2+1))
            plt.title('MSE for y' + str(i+1) +' training prediction', fontsize = s)
            plt.xlabel('k-step ahead', fontsize = s)
            plt.ylabel('MSE', fontsize = s)
            plt.savefig('MSE_train '+str(i+1)+'.png', dpi=600,bbox_inches='tight')        
    
        if train_ratio < 1: 
            for i in range(m_y):
                plt.figure(figsize=(3,2))
                plt.plot(MSE_val[i::m_y], 'd-', color = cmap(i*2+1))
                plt.title('MSE for y' + str(i+1) +' validation prediction', fontsize = s)
                plt.xlabel('k-step ahead', fontsize = s)
                plt.ylabel('MSE', fontsize = s)
                plt.savefig('MSE_val '+str(i+1)+'.png', dpi=600,bbox_inches='tight')  

                
        if test:
            for i in range(m_y):
                plt.figure(figsize=(3,2))
                plt.plot(MSE_test[i::m_y], 'd-', color = cmap(i*2+1))
                plt.title('MSE for y' + str(i+1) +' testing prediction', fontsize = s)
                plt.xlabel('k-step ahead', fontsize = s)
                plt.ylabel('MSE', fontsize = s)
                plt.savefig('MSE_test'+str(i+1)+'.png', dpi=600,bbox_inches='tight')

    
    optimal_params = {}
    optimal_params['lag'] = myresults['mylag']
    optimal_params['deg'] = myresults['mydeg']
    optimal_params['ord'] = myresults['ord']
        

        
        
    return(optimal_params, myresults, MSE_train, MSE_val, MSE_test, y_predict_train, y_predict_val, y_predict_test, train_error, val_error, test_error)
    
    
    
    
    
    
    
    
    
    
    
    
    
def Adaptx_matlab_multi(X, y, timeindex, num_series, data_url, url, X_test=None, y_test=None, train_ratio = 1,\
                      mymaxlag = 12, mydegs = [-1, 0, 1], mynow = 1, steps = 10, plot = True):

    '''This function fits the CVA-state space model for training data X, y of first training ratio data,
    and use rest (1-ratio_data) for forcasting. There can also be test data to test the fitted state space model
    Input:
        X: dictionary of training data predictors numpy array: Nxm, composed of all the data (several time seireses)
        y: dictionary of training data response numy array: Nx1, composed of all the data (several time seireses)
        timeindex: time invterval for each seperate time series, stored in one dictionary, labeled by times seires index, which has shape (N,)
        train_ratio: float, portion of training data used to train the model, and the rest is used as validation data, is applied to every time seires
        num_series: total number of time series contained
        X_test: testing data predictors numpy arrray: N_testxm
        y_test: testing data response numpy array: N_test x 1
        data_url: desired working directory for saving all the results, be a sub-folder of the main ADPATX folder
        url: main directory for ADAPTX folder
        mymaxlag: maximum lag number considered for the CVA, default = 12
        mydegs: degs considered for the trend in state space model, can chose from [-1 0 1 2 3,4], defualt [-1 0 1]
        mynow: instantaneous effect of u on y in state space model, 1: yes, 0: no, defualt 1
        steps: number of steps considered for prediction
        plot: flag for plotting results or not, default TRUE
        
        
    Output:
        optimal_params: dictonary
            mylag: int, selected optimal lag number by AIC
            mydeg: int, selected optimal order on trend by AIC
            ord: int, selected optimal order for CVA by AIC
        
        myresults:
            State space model parameters:
                Phi, G, H, A, B, Q, R, E, F, ABR, BBR
            CVA projection: Jk
            
        Preditction results:
            MSE_train for several, MSE_test with differnt prediction steps
            
                Ym: prediction by final optimal model, Num_steps X timeinstances, the frist row is one step ahead by Kalman
                error: the error Ym-Yp

    '''
    cum = 0
    ##scale data
    for i in range(num_series):
        num = np.shape(timeindex[i+1])[0]       
        num_up_to = round(train_ratio*num)
        if i == 0:
            y_scale = y[cum:cum+num_up_to]
            X_scale = X[cum:cum+num_up_to]
        else:
            y_scale = np.vstack((y_scale, y[cum:cum+num_up_to]))
            X_scale = np.vstack((X_scale,X[cum:cum+num_up_to]))
        cum += num

    scaler = StandardScaler()
    scaler.fit(X_scale)
    X = scaler.transform(X)
        
    scalery = StandardScaler()
    scalery.fit(y_scale)
    y=scalery.transform(y)

    ###Save Data in desired form to mydata.mat, which contains mydata as a matrix of size (m_y+m_x)xN, where y as the first row
    timax = 0
    filist = []
    cum = 0
    for i in range(num_series):
        num = np.shape(timeindex[i+1])[0]       
        num_up_to = round(train_ratio*num)
        if timax<num_up_to: timax = num_up_to
        filist.append('filist' + str(i+1))
         
        d = np.vstack((np.transpose(y[cum:cum+num_up_to]),np.transpose(X[cum:cum+num_up_to])))
        sio.savemat(data_url+'filist' + str(i+1)+'.mat', {'d':d, 'timint':timeindex[i+1][:num_up_to]})
        cum += num
        
    sio.savemat('myfilist.mat', {'filist':filist,'timax':timax, 'num_series':num_series})    
    
        
        
    
    if y_test is not None:
        ###Save test Data in desired form to mydataval.mat
        X_test = scaler.transform(X_test)
        y_test = scalery.transform(y_test)
        
        mydataval=np.vstack((np.transpose(y_test),np.transpose(X_test)))    
        sio.savemat('mydataval.mat', {'mydataval':mydataval})
        test = 1
    else:
        test = 0
    
    
    m_y = np.shape(y)[1]
    m_u = np.shape(X)[1]
    ###Save parameters in a file
    sio.savemat('myparams.mat', {'url':url,'data_url':data_url, 'mydimy':m_y, 'mydimu':m_u, 'mymaxlag':mymaxlag,'mydegs':mydegs, 'mynow':mynow, 'val':test, 'steps':steps})
     
    
    ###Call the matlab script
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())
    #eng.addpath(url, nargout=0)
    eng.CVA_multiset_fit_test(nargout=0)
    
    
    ###Read Results and do Plots
    #the parameters are saved in myresults
    myresults = sio.loadmat(data_url+'myresults.mat')
    
    
    '''Do prediction, first for the training and validation data (if train_ratio<1), for each time series'''
    cum=0
    for i in range(num_series):
        sio.savemat('myparams_prediction.mat', {'url':url,'data_url':data_url, 'steps':steps, 'id':i})
        
        num = np.shape(timeindex[i+1])[0]       
         
        mydataval_prediction = np.vstack((np.transpose(y[cum:cum+num]),np.transpose(X[cum:cum+num])))
        sio.savemat('mydataval_prediction.mat', {'mydataval_prediction':mydataval_prediction})

        cum += num        
        
        eng.cd(os.getcwd())
        eng.CVA_prediction(nargout=0)
        
    eng.quit()



    '''load prediction restuls for training and validation'''
    y_real_train = {}
    y_real_val = {}
    y_predict_train = {}
    y_predict_val = {}
    train_error = {}
    val_error = {}
    
    MSE_train = np.zeros((num_series, steps*m_y))
    MSE_val = np.zeros((num_series, steps*m_y))
    
    for i in range(num_series):
        prediction_train = sio.loadmat(data_url+'kstep' + str(i) + '.mat')
        
        num = np.shape(timeindex[i+1])[0]       
        n_train = round(train_ratio*num)
        
        
        y_real_train[i+1] = np.array(prediction_train['yp'])
        if train_ratio < 1:
            y_real_val[i+1] = y_real_train[i+1][:,n_train:]
            y_real_train[i+1] = y_real_train[i+1][:,:n_train]
        
        y_predict_train[i+1] = np.array(prediction_train['ym'])
        if train_ratio < 1:
            y_predict_val[i+1] = y_predict_train[i+1][:,n_train:]
            y_predict_train[i+1] = y_predict_train[i+1][:,:n_train]
        else:
            y_predict_val[i+1] = None
        
        train_error[i+1] = np.array(prediction_train['ye'])
        if train_ratio < 1:
            val_error[i+1] = train_error[i+1][:,n_train:]
            train_error[i+1] = train_error[i+1][:,:n_train]
            MSE_val[i] = np.nansum(val_error[i+1]**2,axis=1)/np.sum(~np.isnan(val_error[i+1]),axis=1)
        else:
            MSE_val[i] = None
            val_error[i+1] = None
            
        MSE_train[i] = np.nansum(train_error[i+1]**2,axis=1)/np.sum(~np.isnan(train_error[i+1]),axis=1)
                   
        
    '''Prediction for testing data is done already if y_test is not none'''
    if test:
        prediction_test = sio.loadmat(data_url+'kstep_testing.mat')
        y_real_test = np.array(prediction_test['yp'])
        y_predict_test = np.array(prediction_test['ym'])
        test_error = np.array(prediction_test['ye'])    
        MSE_test = np.nansum(test_error**2,axis=1)/np.sum(~np.isnan(test_error),axis=1)
    else:
        MSE_test = None
        test_error = None
        y_predict_test = None   
        
    

    
    #plot the prediction results
    if plot: 
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
            
        s=12

        #plot the prediction vs real
        for i in range(steps):
            for j in range(m_y):
                
                for index in range(num_series):
                    plt.figure(figsize=(5,3))
                    plt.plot(y_real_train[index+1][j], color= cmap(j*2+1), label= 'real')
                    plt.plot(y_predict_train[index+1][m_y*i+j], '--', color= 'xkcd:coral', label = 'prediction')
                    plt.title('Training data' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                    plt.xlabel('Time index',fontsize=s)
                    plt.ylabel('y',fontsize=s)
                    plt.legend(fontsize=s)
                    plt.tight_layout()                    
                    plt.savefig('Train_var_' + str(j+1)+'_step_'+str(i+1)+ '_series_' + str(index+1) +'.png', dpi = 600,bbox_inches='tight')
                    
                    if train_ratio < 1:
                        plt.figure(figsize=(5,3))
                        plt.plot(y_real_val[index+1][j], color= cmap(j*2+1), label= 'real')
                        plt.plot(y_predict_val[index+1][m_y*i+j], '--', color= 'xkcd:coral',label = 'prediction')
                        plt.title('Validation data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                        plt.xlabel('Time index',fontsize=s)
                        plt.ylabel('y',fontsize=s)
                        plt.legend(fontsize=s)
                        plt.tight_layout()                    
                        plt.savefig('Val_var_' + str(j+1)+'_step_'+str(i+1)+ '_series_' + str(index+1) + '.png', dpi = 600,bbox_inches='tight')

                if test:
                    plt.figure(figsize=(5,3))
                    plt.plot(y_real_test[j], color= cmap(j*2+1), label= 'real')
                    plt.plot(y_predict_test[m_y*i+j], '--',color= 'xkcd:coral', label = 'prediction')
                    plt.title('Test data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                    plt.xlabel('Time index',fontsize=s)
                    plt.ylabel('y',fontsize=s)
                    plt.legend(fontsize=s)
                    plt.tight_layout()                    
                    plt.savefig('Test_var_' + str(j+1)+'_step_'+str(i+1)+'.png', dpi = 600,bbox_inches='tight')

                
#                plt.close('all')
        
        #plot fitting errors
        for index in range(num_series):
            max_limit=np.nanmax(train_error[index+1][-m_y:],axis=1)
            min_limit=np.nanmin(train_error[index+1][-m_y:],axis=1)
            
            fig, axs = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
            
            if m_y >1:
                for i in range(steps):
                    for j in range(m_y):
                        axs[i,j].plot(train_error[index+1][m_y*i+j], color= cmap(j*2+1))
                        axs[i,j].set_title('Training data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                        axs[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                        if i is steps-1:
                            axs[i,j].set_xlabel('Time index', fontsize=s)              
                fig.tight_layout()
                plt.savefig('Train error series ' + str(index+1) +'.png', dpi = 600,bbox_inches='tight')
            else:
                j=0
                for i in range(steps):
                    axs[i].plot(train_error[index+1][m_y*i+j], color= cmap(j*2+1))
                    axs[i].set_title('Training data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs[i].set_ylim(min_limit*1.5,max_limit*1.5)
                    if i is steps-1:
                        axs[i].set_xlabel('Time index', fontsize=s)              
                fig.tight_layout()
                plt.savefig('Train error series ' + str(index+1) +'.png', dpi = 600,bbox_inches='tight')
                
            
            if train_ratio < 1: 
                
                max_limit=np.nanmax(val_error[index+1][-m_y:],axis=1)
                min_limit=np.nanmin(val_error[index+1][-m_y:],axis=1)
                fig1, axs1 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
                
                if m_y >1:
                    for i in range(steps):
                        for j in range(m_y):
                            axs1[i,j].plot(val_error[index+1][m_y*i+j], color= cmap(j*2+1))
                            axs1[i,j].set_title('Val data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                            axs1[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                            if i is steps-1:
                                axs1[i,j].set_xlabel('Time index', fontsize=s)   
                else:
                    j=0
                    for i in range(steps):
                        axs1[i].plot(val_error[index+1][m_y*i+j], color= cmap(j*2+1))
                        axs1[i].set_title('Val data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                        axs1[i].set_ylim(min_limit*1.5,max_limit*1.5)
                        if i is steps-1:
                            axs1[i].set_xlabel('Time index', fontsize=s)  
                            
                fig1.tight_layout()
                plt.savefig('Val error series ' + str(index+1) +'.png', dpi=600,bbox_inches='tight')
            
            
        if test:  
            max_limit=np.nanmax(test_error[-m_y:],axis=1)
            min_limit=np.nanmin(test_error[-m_y:],axis=1)
            fig2, axs2 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
            
            if m_y >1:
                for i in range(steps):
                    for j in range(m_y):
                        axs2[i,j].plot(test_error[m_y*i+j], color= cmap(j*2+1))
                        axs2[i,j].set_title('Test data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                        axs2[i,j].set_ylim(min_limit[j]*1.5,max_limit[j]*1.5)
                        if i is steps-1:
                            axs2[i,j].set_xlabel('Time index', fontsize=s)     
                        
            else:    
                j=0
                for i in range(steps):
                    axs2[i].plot(test_error[m_y*i+j], color= cmap(j*2+1))
                    axs2[i].set_title('Test data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs2[i].set_ylim(min_limit*1.5,max_limit*1.5)
                    if i is steps-1:
                        axs2[i].set_xlabel('Time index', fontsize=s)    
                        
            fig2.tight_layout()
            plt.savefig('Test error.png', dpi=600,bbox_inches='tight')        
        
        
        #MSE for prediction results over different steps
        for index in range(num_series):
            for i in range(m_y):
                plt.figure(figsize=(3,2))
                plt.plot(MSE_train[index,i::m_y], 'd-', color = cmap(i*2+1))
                plt.title('MSE for y' + str(i+1) +' training prediction', fontsize = s)
                plt.xlabel('k-step ahead', fontsize = s)
                plt.ylabel('MSE', fontsize = s)
                plt.tight_layout()                    
                plt.savefig('MSE_train '+str(i+1)+ ' series '+ str(index+1)+'.png', dpi=600,bbox_inches='tight')        
        
            if train_ratio < 1: 
                for i in range(m_y):
                    plt.figure(figsize=(3,2))
                    plt.plot(MSE_val[index,i::m_y], 'd-', color = cmap(i*2+1))
                    plt.title('MSE for y' + str(i+1) +' validation prediction', fontsize = s)
                    plt.xlabel('k-step ahead', fontsize = s)
                    plt.ylabel('MSE', fontsize = s)
                    plt.tight_layout()                    
                    plt.savefig('MSE_val '+str(i+1) + ' series '+ str(index+1)+'.png', dpi=600,bbox_inches='tight')  

                
        if test:
            for i in range(m_y):
                plt.figure(figsize=(3,2))
                plt.plot(MSE_test[i::m_y], 'd-', color = cmap(i*2+1))
                plt.title('MSE for y' + str(i+1) +' testing prediction', fontsize = s)
                plt.xlabel('k-step ahead', fontsize = s)
                plt.ylabel('MSE', fontsize = s)
                plt.tight_layout()                    
                plt.savefig('MSE_test'+str(i+1)+'.png', dpi=600,bbox_inches='tight')

    optimal_params = {}
    optimal_params['lag'] = myresults['mylag']
    optimal_params['deg'] = myresults['mydeg']
    optimal_params['ord'] = myresults['ord']
        

        
        
    return(optimal_params, myresults, MSE_train, MSE_val, MSE_test, y_predict_train, y_predict_val, y_predict_test, train_error, val_error, test_error)
    
    
    
    




def Adaptx_matlab_prediction(X, y, data_url, url, X_scale= None, y_scale=None, steps = 3, index = 0, plot = True):
    
    '''This function used the fitted model (Stored in data_url)
    Input:
        X: dictionary of training data predictors numpy array: Nxm, composed of all the data (several time seireses)
        y: dictionary of training data response numy array: Nx1, composed of all the data (several time seireses)
        data_url: desired working directory for saving all the results, be a sub-folder of the main ADPATX folder
        url: main directory for ADAPTX folder
        X_scale: used to scale data
        y_scale: used to scale data
        steps: number of steps considered for prediction
        index: saved as kstep(index).mat file
        plot: flag for plotting results or not, default TRUE
        
        
    Output:
        Preditction results:
            MSE_test with differnt prediction steps
            Ym: prediction by final optimal model, Num_steps X timeinstances, the frist row is one step ahead by Kalman
            error: the error Ym-Yp
    '''
    
    #scale data
    if X_scale is not None:
        scaler = StandardScaler()
        scaler.fit(X_scale)
        X = scaler.transform(X)
            
        scalery = StandardScaler()
        scalery.fit(y_scale)
        y=scalery.transform(y)
        
        
        
    #save data and params
    sio.savemat('myparams_prediction.mat', {'url':url,'data_url':data_url, 'steps':steps, 'id':index})
                 
    mydataval_prediction = np.vstack((np.transpose(y),np.transpose(X)))
    sio.savemat('mydataval_prediction.mat', {'mydataval_prediction':mydataval_prediction})

    
    ##load matlab
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())
    #eng.addpath(url, nargout=0)
    eng.CVA_prediction(nargout=0)
    
    eng.quit()
    
    
    
    #load results
    prediction_test = sio.loadmat(data_url+'kstep' + str(index) +'.mat')
    y_real_test = np.array(prediction_test['yp'])
    y_predict_test = np.array(prediction_test['ym'])
    test_error = np.array(prediction_test['ye'])    
    MSE_test = np.nansum(test_error**2,axis=1)/np.sum(~np.isnan(test_error),axis=1)
    
    m_y = np.shape(y)[1]
    
    #plot results
    if plot: 
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
            
        s=12

        #plot the prediction vs real
        for i in range(steps):
            for j in range(m_y):
                plt.figure(figsize=(5,3))
                plt.plot(y_real_test[j], color= cmap(j*2+1), label= 'real')
                plt.plot(y_predict_test[m_y*i+j], '--',color= 'xkcd:coral', label = 'prediction')
                plt.title('Test data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()                    
                plt.savefig('Test_var_' + str(j+1)+'_step_'+str(i+1)+ '_index_' + str(index)+'.png', dpi = 600,bbox_inches='tight')

#                plt.close('all')
                
        
        #plot fitting errors
        max_limit=np.nanmax(test_error[-m_y:],axis=1)
        min_limit=np.nanmin(test_error[-m_y:],axis=1)
        fig2, axs2 = plt.subplots(steps,m_y,figsize=(3*m_y,2*steps))
        
        if m_y >1:
            for i in range(steps):
                for j in range(m_y):
                    axs2[i,j].plot(test_error[m_y*i+j], color= cmap(j*2+1))
                    axs2[i,j].set_title('Test data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs2[i,j].set_ylim(min_limit[j]-abs(min_limit[j])*0.5,max_limit[j]*1.5)
                    if i is steps-1:
                        axs2[i,j].set_xlabel('Time index', fontsize=s)      
        else:
            for i in range(steps):
                axs2[i].plot(test_error[m_y*i], color= cmap(2+1))
                axs2[i].set_title('Test data ' + str(i+1) +'-step error for y' + str(1), fontsize=s)
                axs2[i].set_ylim(min_limit-abs(min_limit)*0.5,max_limit*1.5)
                if i is steps-1:
                    axs2[i].set_xlabel('Time index', fontsize=s)                
        fig2.tight_layout()
        plt.savefig('Test error ' + str(index) +'.png', dpi=600,bbox_inches='tight')        
        
        
        #MSE for prediction results over different steps
        for i in range(m_y):
            plt.figure(figsize=(3,2))
            plt.plot(MSE_test[i::m_y], 'd-', color = cmap(i*2+1))
            plt.title('MSE for y' + str(i+1) +' testing prediction', fontsize = s)
            plt.xlabel('k-step ahead', fontsize = s)
            plt.ylabel('MSE', fontsize = s)
            plt.tight_layout()                    
            plt.savefig('MSE_test_var_'+str(i+1)+'_index_'+str(index)+'.png', dpi=600,bbox_inches='tight')

        
        
    return(MSE_test,  y_predict_test, test_error)
    