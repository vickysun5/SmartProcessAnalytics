# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:22:36 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

'''This file is to use AIC/AICc/BIC to select the hyper parameter for DALVEN and RNN.'''
   
   
from sklearn.feature_selection import VarianceThreshold
import regression_models as rm
import numpy as np
import nonlinear_regression as nr
import timeseries_regression_RNN as RNN

def IC_mse(model_name, X, y, X_test, y_test, X_val =None, y_val = None, cv_type = None, alpha_num =50, eps = 1e-4, round_number = '', **kwargs):
    '''This function determines the best hyper_parameter using mse based on AIC/AICc
    Input:
    model_name: str, indicating which model to use
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    cv_type: 'BIC', 'AIC' or 'AICc', if not specified use the 40 rule of thumb for 'AIC'
    
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
    
    
    if model_name == 'DALVEN':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] =  [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
        
        if cv_type == None:
            if X.shape[0]//X.shape[1]<40:
                cv_type = 'AICc'
            else:
                cv_type = 'AIC'
                
                
        IC_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag'])))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        for k in range(len(kwargs['degree'])):
            for j in range(len(kwargs['l1_ratio'])):
                for i in range(alpha_num):
                    for t in range(len(kwargs['lag'])):
#                        print(k,j,i,t)
                        _, _, _, _, _, _ , _, _, (AIC,AICc,BIC)= DALVEN(X, y, X_test, y_test, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                        if cv_type == 'AICc':
                            IC_result[k,i,j,t] += AICc
                        elif cv_type == 'BIC':
                            IC_result[k,i,j,t] += BIC
                        else:
                            IC_result[k,i,j,t] += AIC

            
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
        lag = kwargs['lag'][ind[3]]
       
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
        if kwargs['label_name'] :
            
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


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, IC_result[ind], list_name_final)


    ###################################################################################################################
    elif model_name == 'DALVEN_full_nonlinear':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2] #,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] =  [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
            
            
        IC_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag'])))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        for k in range(len(kwargs['degree'])):
            for j in range(len(kwargs['l1_ratio'])):
                for i in range(alpha_num):
                    for t in range(len(kwargs['lag'])):
#                        print(k,j,i,t)
                        _, _, _, _, _, _ , _, _, (AIC,AICc,BIC)= DALVEN(X, y, X_test, y_test, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                        if cv_type == 'AICc':
                            IC_result[k,i,j,t] += AICc
                        elif cv_type == 'BIC':
                            IC_result[k,i,j,t] += BIC
                        else:
                            IC_result[k,i,j,t] += AIC

            
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
        lag = kwargs['lag'][ind[3]]
 

       
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,_= DALVEN(X, y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
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


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, IC_result[ind], list_name_final)



    ###################################################################################################################
    #for RNN, only the model archetecture is viewed as hyper-parameter in thie automated version, the other training parameters can be set by kwargs, otw the default value will be used
    if model_name == 'RNN':
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
            if X_val is None:
                kwargs['train_ratio'] = 0.85
            else:
                kwargs['train_ratio'] = 1
        if 'max_checks_without_progress' not in kwargs:
            kwargs['max_checks_without_progress'] = 100
        if 'epoch_before_val' not in kwargs:
            kwargs['epoch_before_val'] = 300
        
        
        #save or not
        if 'location' not in kwargs:
            kwargs['location'] = 'RNNtest'
        
        ######model training
        if cv_type == None:
            if X.shape[0]//X.shape[1]<40:
                cv_type = 'AICc'
            else:
                cv_type = 'AIC'
                
                
        IC_result = np.zeros((len(kwargs['cell_type']),len(kwargs['activation']), len(kwargs['state_size']), len(kwargs['num_layers'])))
        Result = {}
        
        for i in range(len(kwargs['cell_type'])):
            for j in range(len(kwargs['activation'])):
                for k in range(len(kwargs['state_size'])):
                    for t in range(len(kwargs['num_layers'])):
#                        print(i,j,k,t)
                        p_train,p_val, p_test, (AIC,AICc,BIC),train_loss, val_loss, test_loss = RNN.timeseries_RNN_feedback_single_train(X, y, X_test=X_test, Y_test=y_test, X_val = X_val, Y_val=y_val, train_ratio = kwargs['train_ratio'],\
                                                                                         cell_type=kwargs['cell_type'][i],activation = kwargs['activation'][j], state_size = kwargs['state_size'][k],\
                                                                                         batch_size = kwargs['batch_size'], epoch_overlap = kwargs['epoch_overlap'],num_steps = kwargs['num_steps'],\
                                                                                         num_layers = kwargs['num_layers'][t], learning_rate = kwargs['learning_rate'],  lambda_l2_reg=kwargs['lambda_l2_reg'],\
                                                                                         num_epochs = kwargs['num_epochs'], input_prob = kwargs['input_prob'], output_prob = kwargs['output_prob'], state_prob = kwargs['state_prob'],\
                                                                                         input_prob_test =input_prob_test, output_prob_test = output_prob_test, state_prob_test =state_prob_test,\
                                                                                         max_checks_without_progress = kwargs['max_checks_without_progress'],epoch_before_val=kwargs['epoch_before_val'], location= kwargs['location'], plot= False)
                        if cv_type == 'AICc':
                            IC_result[i,j,k,t] += AICc
                        elif cv_type == 'BIC':
                            IC_result[i,j,k,t] += BIC
                        else:
                            IC_result[i,j,k,t] += AIC
                            
                        Result[(i,j,k,t)] = {'prediction_train':p_train,'prediction_val':p_val,'prediction_test':p_test,'train_loss_final':train_loss,'val_loss_final':val_loss,'test_loss_final':test_loss}
        
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        cell_type = kwargs['cell_type'][ind[0]]
        activation = kwargs['activation'][ind[1]]
        state_size = kwargs['state_size'][ind[2]]
        num_layers = kwargs['num_layers'][ind[3]]
       
        Final = Result[(ind[0],ind[1],ind[2],ind[3])]
        prediction_train,prediction_val, prediction_test, AICs, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, X_test=X_test, Y_test=y_test, X_val = X_val, Y_val=y_val, train_ratio = kwargs['train_ratio'],\
                                                                                         cell_type=cell_type,activation = activation , state_size = state_size,\
                                                                                         batch_size = kwargs['batch_size'], epoch_overlap = kwargs['epoch_overlap'],num_steps = kwargs['num_steps'],\
                                                                                         num_layers = num_layers, learning_rate = kwargs['learning_rate'],  lambda_l2_reg=kwargs['lambda_l2_reg'],\
                                                                                         num_epochs = kwargs['num_epochs'], input_prob = kwargs['input_prob'], output_prob = kwargs['output_prob'], state_prob = kwargs['state_prob'],\
                                                                                         input_prob_test =input_prob_test, output_prob_test = output_prob_test, state_prob_test =state_prob_test,\
                                                                                         max_checks_without_progress = kwargs['max_checks_without_progress'],epoch_before_val=kwargs['epoch_before_val'], location= kwargs['location'], plot= True, round_number = round_number)
                        
        
        hyper_params = {}
        hyper_params['cell_type'] = cell_type
        hyper_params['activation'] = activation
        hyper_params['state_size'] = state_size
        hyper_params['num_layers'] = num_layers
        hyper_params['training_params'] = {'batch_size':kwargs['batch_size'],'epoch_overlap':kwargs['epoch_overlap'],'num_steps':kwargs['num_steps'],'learning_rate':kwargs['learning_rate'],'lambda_l2_reg':kwargs['lambda_l2_reg'],'num_epochs':kwargs['num_epochs']}
        hyper_params['drop_out'] = {'input_prob':kwargs['input_prob'],'output_prob':kwargs['output_prob'], 'state_prob':kwargs['state_prob']}
        hyper_params['early_stop'] = {'train_ratio':kwargs['train_ratio'], 'max_checks_without_progress':kwargs['max_checks_without_progress'],'epoch_before_val':kwargs['epoch_before_val']}
        hyper_params['IC_optimal'] = IC_result[ind]
        
        
        return(hyper_params, kwargs['location'], Final['prediction_train'], Final['prediction_val'], Final['prediction_test'], Final['train_loss_final'], Final['val_loss_final'], Final['test_loss_final'])
