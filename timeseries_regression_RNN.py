# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:29:42 2019

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

"""
Load packages and Set reproduceble results
"""
from sklearn.preprocessing import StandardScaler
import RNN_feedback as RNN_fd
import matplotlib.pyplot as plt



# Seed value
seed_value= 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
seed_value += 1

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
seed_value += 1

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
seed_value += 1

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)


def timeseries_RNN_feedback_single_train(X, Y, X_val = None, Y_val = None, X_test=None, Y_test=None, train_ratio = 0.8,\
                                         cell_type='e',activation = 'tanh', state_size = 2,\
                                         batch_size = 1, epoch_overlap = None,num_steps = 10,\
                                         num_layers = 1, learning_rate = 1e-2, lambda_l2_reg = 1e-3,\
                                         num_epochs =200, input_prob = 0.95, output_prob = 0.95, state_prob = 0.95,\
                                         input_prob_test = 1, output_prob_test = 1, state_prob_test = 1,\
                                         max_checks_without_progress = 100,epoch_before_val=50, location='RNN_feedback_0', round_number = '', plot=False):
    '''This function fits RNN_feedback model to training data, using validation data to determine when to stop,
    when test data is given, it is used to choose the hyperparameter, otherwise AIC will be returned based on training data 
    to select the hyper parameter
    
    Input:
        X: training data predictors numpy array: Nxm
        y: training data response numy array: Nx1
        X_test: testing data predictors numpy arrray: N_testxm
        y_test: testing data response numpy array: N_test x 1
        train_ratio: float, portion of training data used to train the model, and the rest is used as validation data
                     if X_val is provided, this value is overrided
        cell_type: str, type of RNN cell, can be either LSTM, GRU, others for BasicRNN, default = basicRNN
        activation: str, type of activation function, can be relu, tanh, sigmoid, linear, default = tanh
        state_size: int, number of states in the model
        batch_size: int, number of batch used in training
        epoch_overlap: None or int, None indicate no overlap between each training patch, int number represnets the space between each path, (e.g. 0 represtns adjacent patch)
        num_steps: int, number of steps of memory used in dyanmic_RNN training
        num_layer: int, number of RNN layer in the system, default = 1
        learning_rate: float, learning rate for Adam, default= 1e-2
        labda_l2_reg: float, regularization weight, <=0 indicate no regularization, default = 1e-3,
        num_epochs: int, maximum number of epochs considered in the system
        intput_prob, output_prob, state_prob: float, (0, 1], the keep probability for dropout during training, default = 0.95
        intput_prob_test, output_prob_test, state_prob_test: float (0,1], the keep probability for dropout during testing, default = 1 (no dropout)
        max_chekcs_without_progress: int, number of epochs in validation does not improve error for early stopping, default = 100
        epoch_before_val: int, number of epochs in training before using validation set to early stop, default = 50
        location: str, name for saving the trained RNN-feedback model 
        plot: Boolean, whether to plot the training results or not
        
        
    Output:
        (AIC or test results, prediction_train, prediction_test)
    
    '''

    print('========= Loading data =========')
    """
    Load and arrange data for regression
    """
    #parameter for the data sets
    if X_val is None:
        num_train = round(X.shape[0]*train_ratio)
    else:
        num_train = X.shape[0]
    
    if X_test is not None:
        test = True
        num_test = X_test.shape[0]
    else:
        test = False
        
    x_num_features = X.shape[1]
    y_num_features = Y.shape[1]

    print('======== Pre-process Data =========')
    if X_val is None:
        scaler = StandardScaler()
        scaler.fit(X[:num_train])
        X_train = scaler.transform(X[:num_train])
        X_val = scaler.transform(X[num_train:])    
        
        scalery = StandardScaler()
        scalery.fit(Y[:num_train])
        Y_train=scalery.transform(Y[:num_train])
        Y_val = scalery.transform(Y[num_train:])
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)
        X_val = scaler.transform(X_val) 
        
        scalery = StandardScaler()
        scalery.fit(Y)
        Y_train=scalery.transform(Y)
        Y_val = scalery.transform(Y_val)
        
    if test:
        X_test = scaler.transform(X_test)
        Y_test = scalery.transform(Y_test)
    

    input_size_x = x_num_features
    input_size_y = y_num_features
    
    
    
    print('======== Training =========')
    g_train=RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                           num_steps=num_steps, num_layers=num_layers, input_size_x=input_size_x,
                                                           input_size_y=input_size_y , learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg)
    
    train_loss,val_loss,num_parameter = RNN_fd.train_rnn(X_train,Y_train,X_val,Y_val, 
                                                      g_train ,num_epochs, num_steps, batch_size, input_prob, output_prob, state_prob, 
                                                      verbose=True, save=location, epoch_overlap=epoch_overlap, max_checks_without_progress=max_checks_without_progress,
                                                      epoch_before_val = epoch_before_val)
        
    if train_loss is None:
        return (None, None, None, (100000,100000,100000), 100000,100000,100000)
    
    
    
    val_loss = np.array(val_loss)
    if plot:
        '''Plot the result'''
        plt.figure()
        s = 12
        plt.plot(train_loss, color='xkcd:sky blue', label = 'train loss')
        plt.plot(np.linspace(epoch_before_val-1,epoch_before_val+val_loss.shape[0]-1, num = val_loss.shape[0]), val_loss, color= 'xkcd:coral', label = 'val loss')
        plt.title('Traingin and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('# of epoch')
        plt.legend(fontsize=s)
        plt.tight_layout()
        plt.savefig('Training and validation error round ' + round_number +'.png', dpi = 600,bbox_inches='tight')
                 

    ############################################################################
    """Training Final Results"""
    g_train_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                  num_steps= num_train , num_layers=num_layers, input_size_x=input_size_x,
                                                  input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
    
    
    
    prediction_train,train_loss_final,_ = RNN_fd.test_rnn(X_train,Y_train, g_train_final, location, input_prob_test, output_prob_test, state_prob_test, num_train)
    
    AIC = num_train*np.log(np.sum(train_loss_final)/y_num_features) + 2*num_parameter
    AICc = num_train*np.log(np.sum(train_loss_final)/y_num_features) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
    BIC = num_train*np.log(np.sum(train_loss_final)/y_num_features) +  + num_parameter*np.log(num_train)



    ############################################################################
    """Validation Final Results"""
    g_val_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                  num_steps= X_val.shape[0] , num_layers=num_layers, input_size_x=input_size_x,
                                                  input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
    
    
    
    prediction_val,val_loss_final,_ = RNN_fd.test_rnn(X_val,Y_val, g_val_final, location, input_prob_test, output_prob_test, state_prob_test, X_val.shape[0])

    ###############################################for other test sets 0 step
    """Testing Results"""
    if test:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                      num_steps= num_test , num_layers=num_layers, input_size_x=input_size_x,
                                                      input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
    
        
        prediction_test, test_loss_final,_ = RNN_fd.test_rnn(X_test,Y_test, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test)
    else:
        prediction_test = None
        test_loss_final = None    

    
    
    
    #############################################plot training results
    if plot:
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
                
        #plot the prediction vs real
        for j in range(y_num_features):
        
            plt.figure(figsize=(5,3))
            plt.plot(Y_train[1:, j], color= cmap(j*2+1), label= 'real')
            plt.plot(prediction_train[1:, j], '--', color= 'xkcd:coral', label = 'prediction')
            plt.title('RNN Training data prediction for y' + str(j+1),fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y',fontsize=s)
            plt.legend(fontsize=s)
            plt.tight_layout()
            plt.savefig('RNN_train_var_' + str(j+1)+'.png', dpi = 600,bbox_inches='tight')
                
            plt.figure(figsize=(5,3))
            plt.plot(Y_val[1:, j], color= cmap(j*2+1), label= 'real')
            plt.plot(prediction_val[1:, j], '--', color= 'xkcd:coral',label = 'prediction')
            plt.title('RNN Validation data prediction for y' + str(j+1),fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y',fontsize=s)
            plt.legend(fontsize=s)
            plt.tight_layout()                    
            plt.savefig('RNN_val_var_' + str(j+1)+' round ' + round_number +'.png', dpi = 600,bbox_inches='tight')

            if test:
                plt.figure(figsize=(5,3))
                plt.plot(Y_test[1:, j], color= cmap(j*2+1), label= 'real')
                plt.plot(prediction_test[1:, j], '--',color= 'xkcd:coral', label = 'prediction')
                plt.title('RNN Test data prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()                    
                plt.savefig('RNN_test_var_' + str(j+1) + ' round ' + round_number + '.png', dpi = 600,bbox_inches='tight')

                

      
        #plot fitting errors
        for j in range(y_num_features):
        
            plt.figure(figsize=(5,3))
            plt.plot(prediction_train[1:,j]-Y_train[1:,j], color= cmap(j*2+1))
            plt.title('RNN Training error for y' + str(j+1),fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y_pre - y',fontsize=s)
            plt.tight_layout()
            plt.savefig('RNN_train_var_'+ str(j+1)+' error.png', dpi = 600,bbox_inches='tight')
                
            plt.figure(figsize=(5,3))
            plt.plot(prediction_val[1:,j]-Y_val[1:,j], color= cmap(j*2+1))
            plt.title('RNN Validation error for y' + str(j+1),fontsize=s)
            plt.xlabel('Time index',fontsize=s)
            plt.ylabel('y_pre - y',fontsize=s)
            plt.tight_layout()                    
            plt.savefig('RNN_val_var_' + str(j+1)+' round ' + round_number +' error.png', dpi = 600,bbox_inches='tight')

            if test:
                plt.figure(figsize=(5,3))
                plt.plot(prediction_test[1:,j]-Y_test[1:,j], color= cmap(j*2+1))
                plt.title('RNN Test error for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y_pre - y',fontsize=s)
                plt.tight_layout()                    
                plt.savefig('RNN_test_var_' + str(j+1) +' round ' + round_number +' error.png', dpi = 600,bbox_inches='tight')

                

   
    
    return (prediction_train,prediction_val, prediction_test, (AIC,AICc,BIC), train_loss_final, val_loss_final, test_loss_final)















def timeseries_RNN_feedback_multi_train(X, Y, X_val, Y_val, timeindex_train, timeindex_val, X_test=None, Y_test=None,\
                                         cell_type='e',activation = 'tanh', state_size = 2,\
                                         batch_size = 1, epoch_overlap = None,num_steps = 10,\
                                         num_layers = 1, learning_rate = 1e-2, lambda_l2_reg = 1e-3,\
                                         num_epochs =200, input_prob = 0.95, output_prob = 0.95, state_prob = 0.95,\
                                         input_prob_test = 1, output_prob_test = 1, state_prob_test = 1,\
                                         max_checks_without_progress = 100,epoch_before_val=50, location='RNN_feedback_0', plot= False):
    '''This function fits RNN_feedback model to training data, using validation data to determine when to stop,
    when test data is given, it is used to choose the hyperparameter, otherwise AIC will be returned based on training data 
    to select the hyper parameter
    
    Input:
        X: training data predictors numpy array: Nxm
        y: training data response numy array: Nx1
        timeindex: dictionary, starting from 1, each contanis the time index for that seires
        X_test: testing data predictors numpy arrray: N_testxm
        y_test: testing data response numpy array: N_test x 1
        train_ratio: float, portion of training data used to train the model, and the rest is used as validation data
       
        cell_type: str, type of RNN cell, can be either LSTM, GRU, others for BasicRNN, default = basicRNN
        activation: str, type of activation function, can be relu, tanh, sigmoid, linear, default = tanh
        state_size: int, number of states in the model
        batch_size: int, number of batch used in training
        epoch_overlap: None or int, None indicate no overlap between each training patch, int number represnets the space between each path, (e.g. 0 represtns adjacent patch)
        num_steps: int, number of steps of memory used in dyanmic_RNN training
        num_layer: int, number of RNN layer in the system, default = 1
        learning_rate: float, learning rate for Adam, default= 1e-2
        labda_l2_reg: float, regularization weight, <=0 indicate no regularization, default = 1e-3,
        num_epochs: int, maximum number of epochs considered in the system
        intput_prob, output_prob, state_prob: float, (0, 1], the keep probability for dropout during training, default = 0.95
        intput_prob_test, output_prob_test, state_prob_test: float (0,1], the keep probability for dropout during testing, default = 1 (no dropout)
        max_chekcs_without_progress: int, number of epochs in validation does not improve error for early stopping, default = 100
        epoch_before_val: int, number of epochs in training before using validation set to early stop, default = 50
        location: str, name for saving the trained RNN-feedback model 
    
    Output:
        (AIC or test results, prediction_train, prediction_test)
    
    '''

    print('========= Loading data =========')
    """
    Load and arrange data for regression
    """
    #parameter for the data sets    
    if X_test is not None:
        test = True
        num_test = X_test.shape[0]
    else:
        test = False
        
    x_num_features = X.shape[1]
    y_num_features = Y.shape[1]

    print('======== Pre-process Data =========')
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    X_val = scaler.transform(X_val)
    if test:
        X_test = scaler.transform(X_test)
    
    scalery = StandardScaler()
    scalery.fit(Y)
    Y_train=scalery.transform(Y)
    Y_val = scalery.transform(Y_val)
    if test:
        Y_test=scalery.transform(Y_test)
    
    
    input_size_x = x_num_features
    input_size_y = y_num_features
    
    
    
    print('======== Training =========')
    g_train=RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                           num_steps=num_steps, num_layers=num_layers, input_size_x=input_size_x,
                                                           input_size_y=input_size_y , learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg)
    
    train_loss,val_loss,num_parameter = RNN_fd.train_rnn_multi(X,Y,X_val,Y_val, timeindex_train, timeindex_val,
                                                      g_train ,num_epochs, num_steps, batch_size, input_prob, output_prob, state_prob, 
                                                      verbose=True, save=location, epoch_overlap=epoch_overlap, max_checks_without_progress=max_checks_without_progress,
                                                     epoch_before_val = epoch_before_val)
    
    '''Plot the result'''
    s = 12
    val_loss= np.array(val_loss)
    plt.plot(train_loss, color='xkcd:sky blue', label = 'train loss')
    plt.plot(np.linspace(epoch_before_val-1,epoch_before_val+val_loss.shape[0]-1, num = val_loss.shape[0]), val_loss, color= 'xkcd:coral', label = 'val loss')
    plt.title('Traingin and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('# of epoch')
    plt.legend(fontsize=s)
    plt.tight_layout()
    plt.savefig('Training and validation error.png', dpi = 600,bbox_inches='tight')
                 
    
    ############################################################################
    cum = 0
    train_loss_final = []
    prediction_train = []
    train_loss = []
    
    """Training Final Results"""
    for index in range(len(timeindex_train)):
        num = np.shape(timeindex_train[index+1])[0] 
        
        g_train_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                  num_steps= num , num_layers=num_layers, input_size_x=input_size_x,
                                                  input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
        
        

    
        train_pred,loss,_ = RNN_fd.test_rnn(X_train[cum:cum+num],Y_train[cum:cum+num], g_train_final, location, input_prob_test, output_prob_test, state_prob_test, num)
              
        prediction_train.append(train_pred)
        train_loss.append(loss*num)
        train_loss_final.append(loss)
        
        
        if plot:
            import matplotlib
            cmap = matplotlib.cm.get_cmap('Paired')
            

            #plot the prediction vs real
            for j in range(y_num_features):
            
                plt.figure(figsize=(5,3))
                plt.plot(Y_train[cum+1:cum+num,j], color= cmap(j*2+1), label= 'real')
                plt.plot(train_pred[1:,j], '--', color= 'xkcd:coral', label = 'prediction')
                plt.title('RNN Training data prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN_train_var_' + str(j+1)+'.png', dpi = 600,bbox_inches='tight')
                    
      
          
            #plot fitting errors
            for j in range(y_num_features):
            
                plt.figure(figsize=(5,3))
                plt.plot(train_pred[1:,j]-Y_train[cum+1:cum+num,j], color= cmap(j*2+1))
                plt.title('Training error for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y_pre - y',fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN_train_var_' + str(j+1)+' error.png', dpi = 600,bbox_inches='tight')

        
        cum += num

        
    AIC = cum*np.log(np.sum(train_loss)/cum/y_num_features) + 2*num_parameter
    AICc = cum*np.log(np.sum(train_loss)/cum/y_num_features) + (num_parameter+cum)/(1-(num_parameter+2)/cum)
    BIC = cum*np.log(np.sum(train_loss)/cum/y_num_features) + np.log(cum)*num_parameter






    ############################################################################
    cum = 0
    prediction_val = []
    val_loss_final = []
    """Validation Final Results"""
    for index in range(len(timeindex_val)):
        num = np.shape(timeindex_val[index+1])[0] 
        
        g_val_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                  num_steps= num , num_layers=num_layers, input_size_x=input_size_x,
                                                  input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
    
    
        val_pred,loss,_ = RNN_fd.test_rnn(X_val[cum:cum+num],Y_val[cum:cum+num], g_val_final, location, input_prob_test, output_prob_test, state_prob_test, num)
              
        prediction_val.append(val_pred)
        val_loss_final.append(loss)
        
        if plot:
            import matplotlib
            cmap = matplotlib.cm.get_cmap('Paired')
            

            #plot the prediction vs real
            for j in range(y_num_features):
                
                plt.figure(figsize=(5,3))
                plt.plot(Y_val[cum+1:cum+num, j], color= cmap(j*2+1), label= 'real')
                plt.plot(val_pred[1:,j], '--', color= 'xkcd:coral', label = 'prediction')
                plt.title('RNN Validation data prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN_val_var_' + str(j+1)+ 'index' + str(index+1)+'.png', dpi = 600,bbox_inches='tight')
                    
      
          
            #plot fitting errors
            for j in range(y_num_features):
            
                plt.figure(figsize=(5,3))
                plt.plot(val_pred[1:,j]-Y_val[cum+1:cum+num ,j], color= cmap(j*2+1))
                plt.title('RNN Validation error for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y_pre - y',fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN_val_var_' + str(j+1)+ 'index' + str(index+1)+' error.png', dpi = 600,bbox_inches='tight')

        
        cum += num

    
    ###############################################for other test sets 0 step
    """Testing Results"""
    if test:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                      num_steps= num_test , num_layers=num_layers, input_size_x=input_size_x,
                                                      input_size_y = input_size_y , learning_rate = learning_rate, lambda_l2_reg=lambda_l2_reg)
    
        
        prediction_test, test_loss_final,_ = RNN_fd.test_rnn(X_test,Y_test, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test)


        if plot:

            #plot the prediction vs real
            for j in range(y_num_features):
            
                plt.figure(figsize=(5,3))
                plt.plot(Y_test[1:,j], color= cmap(j*2+1), label= 'real')
                plt.plot(prediction_test[1:,j], '--', color= 'xkcd:coral', label = 'prediction')
                plt.title('RNN Testing data prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN Test_var_' + str(j+1)+ 'index' + str(index+1)+'.png', dpi = 600,bbox_inches='tight')
                    
      
          
            #plot fitting errors
            for j in range(y_num_features):
            
                plt.figure(figsize=(5,3))
                plt.plot(prediction_test[1:,j]-Y_test[1:,j], color= cmap(j*2+1))
                plt.title('RNN Testing error for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y_pre - y',fontsize=s)
                plt.tight_layout()
                plt.savefig('RNN Test_var_' + str(j+1)+ 'index' + str(index+1) +' error.png', dpi = 600,bbox_inches='tight')






    else:
        prediction_test = None
        test_loss_final = None    

    
    
    return (prediction_train, prediction_val, prediction_test, (AIC,AICc,BIC), train_loss_final, val_loss_final, test_loss_final)








def timeseries_RNN_feedback_test(X, Y, X_test,Y_test, kstep = 1, cell_type='e',activation = 'tanh', state_size = 2,\
                                 num_layers = 1, input_prob_test = 1, output_prob_test = 1, state_prob_test = 1,\
                                 location='RNN_feedback_0', plot=True,round_number = ''):
    '''This function fits RNN_feedback model to training data, using validation data to determine when to stop,
    when test data is given, it is used to choose the hyperparameter, otherwise AIC will be returned based on training data 
    to select the hyper parameter
    
    Input:
        X: training data predictors numpy array: Nxm, used to scale data
        y: training data response numy array: Nx1, used to scale data
        X_test: testing data predictors numpy arrray: N_testxm
        y_test: testing data response numpy array: N_test x 1
        kstep: positive integer for number of steps prediction ahead.The output at time instant t is calculated using previously measured outputs up to time t-K and inputs up to the time instant t.
        
        
        cell_type: str, type of RNN cell, can be either LSTM, GRU, others for BasicRNN, default = basicRNN
        activation: str, type of activation function, can be relu, tanh, sigmoid, linear, default = tanh
        state_size: int, number of states in the model
        num_layer: int, number of RNN layer in the system, default = 1
        num_epochs: int, maximum number of epochs considered in the system
        intput_prob, output_prob, state_prob: float, (0, 1], the keep probability for dropout during training, default = 0.95
        intput_prob_test, output_prob_test, state_prob_test: float (0,1], the keep probability for dropout during testing, default = 1 (no dropout)
        location: str, name for saving the trained RNN-feedback model 
        plot: boolean, plot the figure or not
        
    Output:
        (test results, prediction_train, prediction_test)
    
    '''

    print('========= Loading data =========')
    """
    Load and arrange data for regression
    """
    #parameter for the data sets    
    num_test = X_test.shape[0]

        
    x_num_features = X.shape[1]
    y_num_features = Y.shape[1]

    print('======== Pre-process Data =========')
    scaler = StandardScaler()
    scaler.fit(X)
    X =scaler.transform(X)
    X_test = scaler.transform(X_test)
    
    scalery = StandardScaler()
    scalery.fit(Y)
    Y = scalery.transform(Y)
    Y_test=scalery.transform(Y_test)
    
    
    input_size_x = x_num_features
    input_size_y = y_num_features
    
    
    #################################################
    kstep = kstep-1   #adjustment for the test_rnn code to be comparable with matlab
    
    
    ###############################################k_STEP single layer
    if num_layers == 1:
        """Testing 0 step"""
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                      num_steps= num_test , num_layers=num_layers, input_size_x=input_size_x,
                                                      input_size_y = input_size_y , learning_rate = 0, lambda_l2_reg=0)
        
        
        
        test_y_prediction, test_loss_final ,test_rnn_outputs = RNN_fd.test_rnn(X_test,Y_test, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test)
        
        if kstep > 0:
            """Testing k step"""
            g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                              num_steps= 1 , num_layers=num_layers, input_size_x=input_size_x,
                                                              input_size_y = input_size_y , learning_rate = 0, lambda_l2_reg=0)
                
            test_y_prediction_kstep, test_loss_kstep = RNN_fd.test_rnn_kstep(X_test,Y_test, test_y_prediction,test_rnn_outputs, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test, kstep=kstep)
            
        else:
            test_y_prediction_kstep = None
            test_loss_kstep = None
                
    
    
    ###############################################k_STEP multi layer
    else:
        """Testing 0 step"""
    
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                      num_steps= 1 , num_layers=num_layers, input_size_x=input_size_x,
                                                      input_size_y = input_size_y , learning_rate = 0, lambda_l2_reg=0)
        
        
        
        test_y_prediction, test_loss_final, test_inter_state = RNN_fd.test_rnn_layer(X_test,Y_test, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test, num_layers)
        
    
        if kstep > 0:
            """Testing k step"""
            g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type=cell_type, activation=activation,state_size=state_size,
                                                          num_steps= 1 , num_layers=num_layers, input_size_x=input_size_x,
                                                          input_size_y = input_size_y , learning_rate = 0, lambda_l2_reg=0)
            test_y_prediction_kstep, test_loss_kstep = RNN_fd.test_rnn_kstep_layer(X_test,Y_test, test_y_prediction,test_inter_state, g_test, location, input_prob_test, output_prob_test, state_prob_test, num_test, kstep=kstep)
        
        else:
            test_y_prediction_kstep = None
            test_loss_kstep = None
    
    loss_final = np.vstack((test_loss_final,test_loss_kstep))

    prediction_final = {}
    for i in range(kstep+1):
        if i == 0:
            prediction_final[i+1] = test_y_prediction
        else:
            prediction_final[i+1] = test_y_prediction_kstep[i]
    
    ###############################################plot final
    if plot: 
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Paired')
    
        s=12
        test_prediction_plot = {}
        for i in range(kstep+1):
            if i == 0:
                test_prediction_plot[i] = test_y_prediction
            else:
                test_prediction_plot[i] = test_y_prediction_kstep[i]

        
        if X.shape[0] == X_test.shape[0]:
            if np.sum(X-X_test) < 1e-4:
                name = 'Train' +round_number
            else:
                name = 'Test' +round_number
        else: 
            name = 'Test' + round_number
            
            
        #plot the prediction vs real
        for i in range(kstep+1):
            for j in range(y_num_features):
                plt.figure(figsize=(5,3))
                plt.plot(Y_test[i+1:,j], color= cmap(j*2+1), label= 'real')
                plt.plot(test_prediction_plot[i][1:,j], '--',color= 'xkcd:coral', label = 'prediction')
                plt.title(name+' data ' + str(i+1) +'-step prediction for y' + str(j+1),fontsize=s)
                plt.xlabel('Time index',fontsize=s)
                plt.ylabel('y',fontsize=s)
                plt.legend(fontsize=s)
                plt.tight_layout()                    
                plt.savefig(name+'_var_' + str(j+1)+'_step_'+str(i+1)+ '.png', dpi = 600,bbox_inches='tight')

                
        
        #plot fitting errors
        max_limit=np.max(test_prediction_plot[kstep][kstep+1:],axis=0)
        min_limit=np.min(test_prediction_plot[kstep][kstep+1:],axis=0)
        fig2, axs2 = plt.subplots(kstep+1,y_num_features,figsize=(3*y_num_features,2*(kstep+1)))
        
        if y_num_features >1:
            for i in range(kstep+1):
                for j in range(y_num_features):
                    axs2[i,j].plot(test_prediction_plot[i][1:,j]-Y_test[i+1:,j], color= cmap(j*2+1))
                    axs2[i,j].set_title(name + ' data ' + str(i+1) +'-step error for y' + str(j+1), fontsize=s)
                    axs2[i,j].set_ylim(min_limit[j]-abs(min_limit[j])*0.5,max_limit[j]*1.5)
                    if i is kstep-1:
                        axs2[i,j].set_xlabel('Time index', fontsize=s)      
        else:
            for i in range(kstep+1):
                axs2[i].plot(test_prediction_plot[i][1:]-Y_test[i+1:], color= cmap(2+1))
                axs2[i].set_title(name + ' data ' + str(i+1) +'-step error for y' + str(1), fontsize=s)
                axs2[i].set_ylim(min_limit-abs(min_limit)*0.5,max_limit*1.5)
                if i is kstep-1:
                    axs2[i].set_xlabel('Time index', fontsize=s)                
        fig2.tight_layout()
        plt.savefig(name + ' error kstep.png', dpi=600,bbox_inches='tight')        
        
        
        #MSE for prediction results over different steps
        MSE_test= np.vstack((test_loss_final,test_loss_kstep))
        for i in range(y_num_features):
            plt.figure(figsize=(3,2))
            plt.plot(np.linspace(1,MSE_test.shape[0],num=MSE_test.shape[0]),MSE_test[:,i], 'd-', color = cmap(i*2+1))
            plt.title(name+' MSE for y' + str(i) +' prediction', fontsize = s)
            plt.xlabel('k-step ahead', fontsize = s)
            plt.ylabel('MSE', fontsize = s)
            plt.tight_layout()                    
            plt.savefig('MSE_'+name+'_var_'+str(i)+'.png', dpi=600,bbox_inches='tight')
    
    


    return (prediction_final, loss_final)

