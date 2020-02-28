# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:56:16 2020

@author: Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
(c) 2020 Weike Sun, all rights reserved
"""

import pandas as pd
import numpy as np
from dataset_property_new import nonlinearity_assess, collinearity_assess,  residual_analysis, nonlinearity_assess_dynamic
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
mpl.style.use('default')


import warnings
warnings.filterwarnings('ignore')

"""
Decision Tree for Smart Process Analytics
"""


"""
Create Data Block
"""
print("""----------------------------------------------------
Please provide the following information using 'Yes=1' or 'No=0' 
----------------------------------------------------""")

if_have_testing_data = int(input("Do you have a specific testing dataset you want to test? "))
if_time_series = int(input("The data is time-series? "))
if_interpretable = int(input("Do you require the model to be interpretable? "))
if_continuity = int(input("Do you require the model to be contiuous? e.g., use the model for an optimizer. "))
if_grouped = int(input("Is there any grouping in your measurements? e.g., replicated measurements. "))
if not if_grouped:
    group =None
if_spectra = int(input("Spectral data? "))
if_plot_interrogation = int(input("Do you want to visualize the data interrogation results? "))
if_time = int(input("Do you have time for nested cross-validation? "))
if_robust = int(input("Do you prioritize robustness over accuracy? "))

print('')
print("""----------------------------------------------------
Please provide the following information using 'Yes=1', 'No=0', or 'Unknown=2'
---------------------------------------------------- """)
if_enough = int(input("Do you think you have enough data to characterize the underlying process complexity? "))
if_nonlinear = int(input("Do you believe your model should be nonlinear? For spectral data, the linear model is used. "))
if if_spectra:
    if_nonlinear = 0
    
if_multicollinear= int(input("Do you believe there is siginificant collinearity in your data? "))
if_dynamic = int(input("Do you believe the dynamic model is neccessary? "))

if if_dynamic != 0 and if_time_series == 0:
    print('Warining*Your data is not time series data, so only static model should be considered.*')
    if_dynamic = 0

if_stability = int(input("Do you require information of model stability given enough time? "))

print('')
print("""----------------------------------------------------
Please provide the file name, e.g., mydata.txt:
----------------------------------------------------""")
Data_name = input("Name of your data file (Should have dimension N x (m+1), last column for predicted variable) ")


#load data
if Data_name[-3:] == 'txt':
    if int(input('Is there delimiter (,) in your txt file? ')):
        Data = np.loadtxt(Data_name, delimiter=',')  
    else:
        Data = np.loadtxt(Data_name)
elif Data_name[-3:] == 'lsx':
    Data = pd.read_excel(Data_name, header = None)   
    Data = np.array(Data)
else:
    print("Please convert data type to '.txt' or '.xlsx'")
    
X_original = Data[:,0:-1]
y_original = Data[:,-1].reshape(-1,1)

m = np.shape(X_original)[1]
N = np.shape(X_original)[0]

if if_have_testing_data:
    Test_data_name = input("Please provide test dataset name: ")
    
    if Test_data_name[-3:] == 'txt':
        if int(input('Is there delimiter (,) in your txt file? ')):
            Test_data = np.loadtxt(Test_data_name, delimiter = ',')
        else:
            Test_data = np.loadtxt(Test_data_name)  
    elif Test_data_name[-3:] == 'lsx':
        Test_data = pd.read_excel(Test_data_name, header = None)   
        Test_data = np.array(Test_data)
    else:
        print("Please convert test data type to '.txt' or '.xlsx'")
    
    X_test_original = Test_data[:,0:-1]
    y_test_original = Test_data[:,-1].reshape(-1,1)
    N_test = np.shape(X_test_original)[0]
    
else:
    X_test_original = None
    y_test_original = None
    


"""
Assess Data Property
"""
round_number = 0

print('')
print("""----------------------------------------------------
Please provide the following information to assess data [Yes:1 No:0]:
----------------------------------------------------""")

if int(input("Is there any categorical variable in your X? ")):
    cat = list(map(int,input("Enter a series of indicator for which variable is categorical: e.g., 1 0 0 represents the 1st variable is a categorical variable. ").strip().split()))[:m] 
else:
    cat = None
    
if int(input("Do you have varaibles names you want to use in the figures? [Yes 1 No 0]: ")):
    xticks = list(map(str,input("Enter a series of x variable name: e.g., x1 x2 x3. ").strip().split()))[:m] 
    yticks = list(input("Enter the name of y variable: ").strip().split())[:1] 
else:
    xticks = None
    yticks = None
    
if int(input("Do you have a significance level to use? ")):
    alpha = float(input("Provide the significance level: "))
else:
    alpha = 0.01
        
        
if if_nonlinear == 2:        
    if_nonlinear = nonlinearity_assess(X_original, y_original, if_plot_interrogation, cat = cat,alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number =  round_number)

    if if_nonlinear == 0 and if_dynamic == 1:
        lag = int(input('The lag number you want to use to assess nonlinear dyanmics: '))
        if_nonlinear_dynamic = nonlinearity_assess_dynamic(X_original, y_original, if_plot_interrogation, alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number =  round_number,lag= lag)
    
        if if_nonlinear_dynamic ==1:
            if_nonlinear = 1
    
if if_multicollinear == 2:
    if_multicollinear = int(collinearity_assess(X_original, y_original, if_plot_interrogation, xticks =  xticks, yticks = yticks, round_number = round_number))


"""
Select the Model
"""

if int(input('Do you have a specific model in the data analytical traingle that you want to use? ')):

    model_name = list(map(str,input('Please provide the model name: [ALVEN/SVR/RF/EN/SPLS/RR/PLS or DALVEN or RNN or SS]. Note you can not test static and dynamic model at the same time. If you put in the static model name and then there is significant dynamics in the residual and you did not answer 0 to "do you want a dynamic model", the dyanmic model will be tested automatically after that. ').strip().split()))
    
else:
    print('')
    print("""----------------------------------------------------
Based on the information of data characteristics, the following methods are selected:
----------------------------------------------------""")

    model_name = None
        
    if if_nonlinear == 1:
        if if_dynamic == 0 or if_dynamic == 2:
            print('The nonlinear model is selected:')
            if if_enough == 0:
                print('Because you have limited data, ALVEN is recommonded.')
                model_name = ['ALVEN']
            elif if_interpretable == 1:
                print('Because you would like an interpretable model, ALVEN is recommonded.')
                model_name = ['ALVEN']
            elif if_continuity == 1:
                print('Because you ask for continuity, ALVEN/SVR are recommonded.')
                model_name = ['ALVEN','SVR']
            else:
                print('The nonlinear models, ALVEN/SVR/RF, will be tested.')
                model_name = ['ALVEN','SVR','RF']
        
        if if_dynamic == 1:
            print('The nonlinear dynamic model is selected:')
            if if_enough == 0 :
                print('Because you have limited data, DALVEN is recommonded.')
                model_name = ['DALVEN']
            elif if_interpretable == 1:
                print('Because you would like an interpretable model, DALVEN is recommonded.')
                model_name = ['DALVEN']            
            else:
                print('Because you have engough data and do not require interpretability, RNN is recommonded.')
                model_name = ['RNN']
        
    else:
        if if_dynamic == 0 or if_dynamic ==2:
            if if_multicollinear == 0:
                print('There is no significant nonlinearity and multicollinearity in the data, OLS is recommonded.')
                model_name = ['OLS']
            else:
                if if_spectra:
                    print('Because you have spectral data, RR/PLS are recommonded.')
                    model_name = ['RR','PLS']
                elif if_interpretable:
                    print('Because you want interpretability, EN/SOLS are recommonded.')
                    model_name = ['EN','SPLS']
                else:
                    print('There is significant multicollinearity, EN/SPLS/RR/PLS are recommonded.')
                    model_name = ['EN','SPLS','RR','PLS']
        else:
            print('There is significant dynamics and multicolinearity, CVA/SSARX/MOSEP are recommonded.')
            model_name = ['SS']


"""
Select Cross-Validation Strategy
"""
nested_flag = 0
cv_method = None
one_std = 0
grouped = 0


if 'OLS' in model_name:
    print("")
    print('OLS is selected, no cross-validation is needed.')
    

elif int(input('Do you have a specific cross-validation method that you want to use? ')):

    cv_method = input('Please provide the name of cross-validation method: ')
    if if_grouped:
        grouped = 1
        print('Since you have grouped data, please provide the group lable of each variable in a Nx1 format.')
        group_name = input('Please provide the file name of the label: (e.g. label.txt) ')
        if group_name[-3:] == 'txt':
            if int(input('Is there delimiter (,) in your txt file? ')):
                group = np.loadtxt(group_name, delimiter=',')  
            else:
                group = np.loadtxt(group_name)
        elif group_name[-3:] == 'lsx':
            group = pd.read_excel(group_name, header = None)   
            group = np.array(group)
        else:
            print("Please convert data type to '.txt' or '.xlsx'")

    if if_robust==1:
        one_std = 1
    if if_stability ==1:
        nested_flag = 1

else:
    print('')
    print("""----------------------------------------------------
Based on the information of data attributes, the cross-validation strategy is selected as:
----------------------------------------------------""")
    
    if if_dynamic != 1:
        if not if_grouped:
            if if_enough == 1:
                cv_method = 'Single'
                print('Single validation set is used.')
            else:
                if if_time == 1:
                   if if_stability == 1:
                       if if_robust == 1:
                           cv_method = 'Re_KFold'
                           nested_flag = 1                       
                           one_std = 1
                           print('Nested CV with repeated KFold in inner loop and one-std rule is selected')
                       else:
                           cv_method = 'Re_KFold'
                           nested_flag = 1
                           print('Nested CV with repeated KFold in inner loop is selected')
                   else:
                       if if_robust == 1:
                           if int(input('Do you want to select beweent KFold/MC/Re_KFold? default: Re_Kfold. ')):
                               cv_method = input('Plese provide the CV method name: ')
                               one_std = 1
                               print(cv_method + ' with one-std rule is selected.')
                           else:
                               one_std = 1
                               cv_method = 'Re_KFold'
                               print('Repeated KFold with one-std rule is selected.')
                       else:
                           if int(input('Do you want to select beweent KFold/MC/Re_KFold? default: Re_Kfold. ')):
                               cv_method = input('Plese provide the CV method name: ')
                               print(cv_method + ' is selected.')
                           else:
                               cv_method = 'Re_KFold'
                               print('Repated KFold is selected.')
                               
                else:
                    if if_robust == 1:
                        if int(input('Do you want to select beweent KFold/MC/Re_KFold? default: Re_Kfold. ')):
                           cv_method = input('Plese provide the CV method name: ')
                           one_std = 1
                           print(cv_method + ' with one-std rule is selected.')
                        else:
                           one_std = 1
                           cv_method = 'Re_KFold'
                           print('Repeated KFold with one-std rule is selected.')
                    else:
                        if int(input('Do you want to select beweent KFold/MC/Re_KFold? default: Re_Kfold. ')):
                           cv_method = input('Plese provide the CV method name: ')
                           print(cv_method + ' is selected.')
                        else:
                           cv_method = 'Re_KFold'
                           print('Repated KFold is selected.')
                         
                           
        else:
            grouped = 1
            print('Since you have grouped data, please provide the group lable of each variable in a Nx1 format.')
            group_name = input('Please provide the file name of the label: (e.g. label.txt) ')
            if group_name[-3:] == 'txt':
                if int(input('Is there delimiter (,) in your txt file? ')):
                    group = np.loadtxt(group_name, delimiter=',')  
                else:
                    group = np.loadtxt(group_name)
            elif group_name[-3:] == 'lsx':
                group = pd.read_excel(group_name, header = None)   
                group = np.array(group)
            else:
                print("Please convert data type to '.txt' or '.xlsx'")
            
            
            if if_enough == 1:
                cv_method = 'Single_group'
                print('Single grouped CV is selected')
                
            else:
                if if_time == 1:
                    if if_stability == 1:
                        if if_robust == 1:
                            one_std = 1
                            nested_flag = 1
                            cv_method = 'GroupKFold'
                            print('Nested grouped Kfold with one std rule is selected.')
                        else:
                            nested_flag = 1
                            cv_method = 'GroupKFold'
                            print('Nested grouped Kfold is selected.')
                    else:
                        if if_robust == 1:
                            one_std = 1
                            cv_method = 'GroupKFold'
                            print('Grouped Kfold with one std rule is selected.')
                        else:
                            cv_method = 'GroupKFold'
                            print('Grouped Kfold is selected.')                       
                
                else:
                   if if_robust == 1:
                       one_std = 1
                       cv_method = 'GroupKFold'
                       print('Grouped Kfold with one std rule is selected.')
                   else:
                       cv_method = 'GroupKFold'
                       print('Grouped Kfold is selected.')
    else:
        if model_name == ['SS']:
            print('MATLAB/ADAPTx packges with information criterion will be used')
        
        else:    
            if if_enough == 1:
                cv_method = 'Single_ordered'
                print('Single validation is used for time series modeling.')
            else:
                if if_time:
                    if if_robust:
                        cv_method = 'Timeseries'
                        one_std = 1
                        print('Cross-validation for time series with one std rule is selected.')
                                           
                    else:
                        cv_method = 'Timeseries'
                        print('Cross-validation for time series is selected.')
                else:
                    cv_method = 'IC'
                    print('Information criteria is selected.')
                    if if_robust:
                        one_std = 1
            

                
                



print('')
print("""----------------------------------------------------
Start Model Fitting
----------------------------------------------------""")
round_number = 1
from copy import deepcopy

########################### data preprocessing
X=deepcopy(X_original)
y=deepcopy(y_original)

scaler_x = StandardScaler(with_mean=True, with_std=True)
scaler_x.fit(X)
X_scale = scaler_x.transform(X)

scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y)
y_scale = scaler_y.transform(y)

if X_test_original is not None:
    X_test = deepcopy(X_test_original)
    y_test = deepcopy(y_test_original)
    X_test_scale = scaler_x.transform(X_test)
    y_test_scale = scaler_y.transform(y_test) 
else:
    X_test = X
    y_test = y
    X_test_scale =X_scale
    y_test_scale =y_scale


############################ model fitting 1st round
fitting_result = {}


if 'OLS' in model_name:
    from regression_models import OLS_fitting
    final_model, model_params, mse_train, mse_test, yhat_train, yhat_test = OLS_fitting(X_scale, y_scale, X_test_scale, y_test_scale, 0)
    fitting_result['OLS'] = {'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test}
    selected_model = 'OLS'
   
    yhat_test = scaler_y.inverse_transform(yhat_test)
    _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
    
    if if_dynamic != 0:
        print('The first round static fitting is done, check if nonlinear model is neccesarry')
        if if_dynamic:
            print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
            round_number = 2
        else:
            print('--------------Analysis Is Done--------------')
    else:
        print('--------------Analysis Is Done--------------') 
        
        

elif if_dynamic == 2:
    if not one_std:
        #use static cross-validation for this round and traditional cv
        import cv_final as cv
        
        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
        alpha_num = int(input('Number of penalty weight you want to consider in RR/EN/ALVEN, if not known input 20: '))
        
        
        if not nested_flag:
            print('------Model Construction------')
            
            val_err = np.zeros(len(model_name))
            index = 0
            fitting1_result_trial = {}
            
            for model_index in model_name:
            
                if model_index == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                    val_err[index] = MSE_val
                    
                elif model_index == 'SVR' or model_index == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                    val_err[index] = MSE_val
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    val_err[index] = MSE_val
                
                index += 1
                
            if len(model_name) > 1: 
                print('Select the best model from the small candidate pool based on validation error:')
                selected_model = model_name[np.argmin(val_err)]
                print('*****'+selected_model + ' is selected.'+'*****')
                
            else:
                selected_model = model_name[0]
            
            fitting_result[selected_model]=fitting1_result_trial[selected_model]
                
            yhat_test = scaler_y.inverse_transform(fitting_result[selected_model]['yhat_test'])
            _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
            print('The first round static fitting is done, check if nonlinear model is neccesarry')
            if if_dynamic:
                print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                round_number = 2
            else:
                print('--------------Analysis Is Done--------------')


                

        else: 
            print('Nested CV is used and the model selection if necessary is based on testing set in the outter loop')
            
            if not grouped:
                num_outter = int(input('How many number of outter loop you want to use in Nested CV? if not known input 10: '))
                print('------Model Construction------')

                from sklearn.model_selection import train_test_split
                               
                test_nest_err = np.zeros((len(model_name),num_outter))

                for index_out in range(num_outter):
                    X_nest, X_nest_test, y_nest, y_nest_test = train_test_split(X, y, test_size=1/K_fold, random_state= index_out)
                    X_nest_scale, X_nest_scale_test, y_nest_scale, y_nest_scale_test = train_test_split(X_scale, y_scale, test_size=1/K_fold, random_state= index_out)

                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X_nest, y_nest, X_nest_test, y_nest_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                        
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')
                    
                else:
                    selected_model = model_name[0]

                print('Final model fitting')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
                print('The first round static fitting is done, check if nonlinear model is neccesarry')
                if if_dynamic:
                    print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                    round_number = 2
                else:
                    print('--------------Analysis Is Done--------------')
                 

            
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                print('Leave one group out will be used in the outer loop')
                
                print('------Model Construction------')

                test_nest_err = np.zeros((len(model_name), len(np.unique(group))))
                logo = LeaveOneGroupOut()

                index_out = 0
                for train, test in logo.split(X, y.flatten(), groups=group.flatten()):
                    
                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X[train], y[train], X[test], y[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                    index_out +=1 
                    
                    
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')

                    
                else:
                    selected_model = model_name[0]


                print('------Final model fitting-------')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    

                print('The first round static fitting is done, check if nonlinear model is neccesarry')
                if if_dynamic:
                    print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                    round_number = 2
                else:
                    print('--------------Analysis Is Done--------------')
    
    
    
    else:
        #use static cross-validation for this round and traditional cv
        import cv_final_onestd as cv_std
        
        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
        alpha_num = int(input('Number of penalty weight you want to consider in RR/EN/ALVEN, if not known input 20: '))
        
        
        if not nested_flag:
            print('------Model Construction------')

            val_err = np.zeros(len(model_name))
            index = 0
            fitting1_result_trial = {}
            
            for model_index in model_name:
            
                if model_index == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                    val_err[index] = MSE_val
                    
                elif model_index == 'SVR' or model_index == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                    val_err[index] = MSE_val
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    val_err[index] = MSE_val
                
                index += 1
                
            if len(model_name) > 1: 
                print('Select the best model from the small candidate pool based on validation error:')
                selected_model = model_name[np.argmin(val_err)]
                print('*****'+selected_model + ' is selected.'+'*****')
            else:
                selected_model = model_name[0]
            
            fitting_result[selected_model]=fitting1_result_trial[selected_model]
                
            yhat_test = scaler_y.inverse_transform(fitting_result[selected_model]['yhat_test'])
            _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
            
            print('The first round static fitting is done, check if nonlinear model is neccesarry')
            if if_dynamic:
                print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                round_number = 2
            else:
                print('--------------Analysis Is Done--------------')


        else: 
            print('Nested CV is used and the model selection if necessary is based on testing set in the outter loop')
            
            if not grouped:
                num_outter = int(input('How many number of outter loop you want to use in Nested CV? if not known input 10: '))
                print('------Model Construction------')

                from sklearn.model_selection import train_test_split
                               
                test_nest_err = np.zeros((len(model_name),num_outter))

                for index_out in range(num_outter):
                    X_nest, X_nest_test, y_nest, y_nest_test = train_test_split(X, y, test_size=1/K_fold, random_state= index_out)
                    X_nest_scale, X_nest_scale_test, y_nest_scale, y_nest_scale_test = train_test_split(X_scale, y_scale, test_size=1/K_fold, random_state= index_out)

                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X_nest, y_nest, X_nest_test, y_nest_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                        
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')
                    
                else:
                    selected_model = model_name[0]

                print('Final model fitting')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
                print('The first round static fitting is done, check if nonlinear model is neccesarry')
                if if_dynamic:
                    print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                    round_number = 2
                else:
                    print('--------------Analysis Is Done--------------')
                 

            
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                print('Leave one group out will be used in the outer loop')
                
                print('------Model Construction------')

                test_nest_err = np.zeros((len(model_name), len(np.unique(group))))
                logo = LeaveOneGroupOut()

                index_out = 0
                for train, test in logo.split(X, y.flatten(), groups=group.flatten()):
                    
                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X[train], y[train], X[test], y[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                    index_out +=1 
                    
                    
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')

                    
                else:
                    selected_model = model_name[0]


                print('------Final model fitting-------')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                _, if_dynamic = residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    

                print('The first round static fitting is done, check if nonlinear model is neccesarry')
                if if_dynamic:
                    print('There is significant dynamic in the residual, dyanmic model will be fitted in the 2nd round')
                    round_number = 2
                else:
                    print('--------------Analysis Is Done--------------')
        
                
                
                
                
elif if_dynamic == 0:
    
    if not one_std:
        #use static cross-validation for this round and traditional cv
        import cv_final as cv
        
        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
        alpha_num = int(input('Number of penalty weight you want to consider in RR/EN/ALVEN, if not known input 20: '))
        
        
        if not nested_flag:
            print('------Model Construction------')

            val_err = np.zeros(len(model_name))
            index = 0
            fitting1_result_trial = {}
            
            for model_index in model_name:
            
                if model_index == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                    val_err[index] = MSE_val
                    
                elif model_index == 'SVR' or model_index == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                    val_err[index] = MSE_val
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    val_err[index] = MSE_val
                
                index += 1
                
            if len(model_name) > 1: 
                print('Select the best model from the small candidate pool based on validation error:')
                selected_model = model_name[np.argmin(val_err)]
                print('*****'+selected_model + ' is selected.'+'*****')
                
            else:
                selected_model = model_name[0]
                
            fitting_result[selected_model]=fitting1_result_trial[selected_model]
            
            yhat_test = scaler_y.inverse_transform(fitting_result[selected_model]['yhat_test'])
            residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                
            print('--------------Analysis Is Done--------------')

        else: 
            print('Nested CV is used and the model selection if necessary is based on testing set in the outter loop')
            
            if not grouped:
                num_outter = int(input('How many number of outter loop you want to use in Nested CV? if not known input 10: '))
               
                print('------Model Construction------')

                from sklearn.model_selection import train_test_split
                               
                test_nest_err = np.zeros((len(model_name),num_outter))

                for index_out in range(num_outter):
                    X_nest, X_nest_test, y_nest, y_nest_test = train_test_split(X, y, test_size=1/K_fold, random_state= index_out)
                    X_nest_scale, X_nest_scale_test, y_nest_scale, y_nest_scale_test = train_test_split(X_scale, y_scale, test_size=1/K_fold, random_state= index_out)

                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X_nest, y_nest, X_nest_test, y_nest_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                        
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')
                    
                else:
                    selected_model = model_name[0]

                print('Final model fitting')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
                print('--------------Analysis Is Done--------------')
                 

            
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                print('Leave one group out will be used in the outer loop')
                
                print('------Model Construction------')

                test_nest_err = np.zeros((len(model_name), len(np.unique(group))))
                logo = LeaveOneGroupOut()

                index_out = 0
                for train, test in logo.split(X, y.flatten(), groups=group.flatten()):
                    
                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(model_index, X[train], y[train], X[test], y[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                    index_out +=1 
                    
                    
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')

                    
                else:
                    selected_model = model_name[0]


                print('------Final model fitting-------')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    

                print('--------------Analysis Is Done--------------')
    
    
    
    else:
        #use static cross-validation for this round and traditional cv
        import cv_final_onestd as cv_std
        
        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
        alpha_num = int(input('Number of penalty weight you want to consider in RR/EN/ALVEN, if not known input 20: '))
        
        
        if not nested_flag:
            print('------Model Construction------')

            val_err = np.zeros(len(model_name))
            index = 0
            fitting1_result_trial = {}
            
            for model_index in model_name:
            
                if model_index == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                    val_err[index] = MSE_val
                    
                elif model_index == 'SVR' or model_index == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                    val_err[index] = MSE_val
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting1_result_trial[model_index] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    val_err[index] = MSE_val
                
                index += 1
                
            if len(model_name) > 1: 
                print('Select the best model from the small candidate pool based on validation error:')
                selected_model = model_name[np.argmin(val_err)]
                print('*****'+selected_model + ' is selected.'+'*****')
            else:
                selected_model = model_name[0]
    
            fitting_result[selected_model]=fitting1_result_trial[selected_model]
                
            yhat_test = scaler_y.inverse_transform(fitting_result[selected_model]['yhat_test'])
            residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
            
            print('--------------Analysis Is Done--------------')


        else: 
            print('Nested CV is used and the model selection if necessary is based on testing set in the outter loop')
            
            if not grouped:
                num_outter = int(input('How many number of outter loop you want to use in Nested CV? if not known input 10: '))
                
                print('------Model Construction------')

                from sklearn.model_selection import train_test_split
                               
                test_nest_err = np.zeros((len(model_name),num_outter))

                for index_out in range(num_outter):
                    X_nest, X_nest_test, y_nest, y_nest_test = train_test_split(X, y, test_size=1/K_fold, random_state= index_out)
                    X_nest_scale, X_nest_scale_test, y_nest_scale, y_nest_scale_test = train_test_split(X_scale, y_scale, test_size=1/K_fold, random_state= index_out)

                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X_nest, y_nest, X_nest_test, y_nest_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_nest_scale, y_nest_scale, X_nest_scale_test, y_nest_scale_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                        
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')
                    
                else:
                    selected_model = model_name[0]

                print('Final model fitting')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
                print('--------------Analysis Is Done--------------')
                 

            
            else:
                from sklearn.model_selection import LeaveOneGroupOut
                print('Leave one group out will be used in the outer loop')
                
                print('------Model Construction------')

                test_nest_err = np.zeros((len(model_name), len(np.unique(group))))
                logo = LeaveOneGroupOut()

                index_out = 0
                for train, test in logo.split(X, y.flatten(), groups=group.flatten()):
                    
                    index = 0
                    for model_index in model_name:
        
                        if model_index == 'ALVEN':
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(model_index, X[train], y[train], X[test], y[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                            test_nest_err[index,index_out] = mse_test
                        
                        elif model_index == 'SVR' or model_index == 'RF':
                            model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                        else:
                            model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(model_index, X_scale[train], y_scale[train], X_scale[test], y_scale[test], cv_type = cv_method, group = group[train], K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                            test_nest_err[index,index_out] = mse_test
                    
                        index += 1
                    index_out +=1 
                    
                    
                print('The nested CV testing MSE result:')
                import matplotlib.pyplot as plt
                plt.figure()
                pos = [i+1 for i in range(len(model_name))]
                ax=plt.subplot(111)
                plt.violinplot(np.transpose(test_nest_err))
                ax.set_xticks(pos)
                ax.set_xticklabels(model_name)
                ax.set_title('Testing MSE distribution using nested CV')

                        
                if len(model_name) > 1: 
                    print('Select the best model from the small candidate pool based on nested test error:')
                    selected_model = model_name[np.argmin(np.mean(test_nest_err,axis=1))]
                    print('*****'+selected_model + ' is selected.*****')
                    
                else:
                    selected_model = model_name[0]


                print('------Final model fitting-------')

                if selected_model == 'ALVEN':
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val, final_list = cv_std.CV_mse(selected_model, X, y, X_test, y_test, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num, label_name=True)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val, 'final_list':final_list}
                        
                elif selected_model == 'SVR' or selected_model == 'RF':
                    model_hyper,final_model, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val':MSE_val}
                else:
                    model_hyper,final_model, model_params, mse_train, mse_test, yhat_train, yhat_test, MSE_val = cv_std.CV_mse(selected_model, X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, group = group, K_fold = K_fold, Nr= Nr, alpha_num=alpha_num)
                    fitting_result[selected_model] = {'model_hyper':model_hyper,'final_model':final_model, 'model_params':model_params, 'mse_train':mse_train, 'mse_test':mse_test, 'yhat_train':yhat_train, 'yhat_test':yhat_test, 'MSE_val': MSE_val}
                    
                yhat_test = scaler_y.inverse_transform(yhat_test)
                residual_analysis(X_test, y_test, yhat_test, alpha = alpha, round_number = round_number)
                    
                print('--------------Analysis Is Done--------------')
        
                
                
else:  #use dynamic model in the first round
    steps = int(input('Number of steps you want to test for the future prediction? '))
    
    if if_nonlinear == 1:

        if model_name == ['RNN']:
            selected_model = 'RNN'
            import timeseries_regression_RNN as t_RNN
            
            print('Please input the following numbers/types from the smallest to the largest: 1 3 5 or linear relu')
            activation = list(map(str,input("Types of activation function you want to use (e.g. linear, relu, tanh?): ").strip().split()))
            num_layers = list(map(int,input("Numbers of layers you want to test: ").strip().split()))
            state_size = list(map(int,input("Numbers of states you want to test: ").strip().split()))
            cell_type = list(map(str,input("Types of cells you want to use (e.g. regular, LSTM, GRU?): ").strip().split()))

            print('Please provide the following training parameter.')
            batch_size = int(input("Number of batch used in training you want to test: "))
            epoch_overlap = input('The overlap between different batch? The number indicate the space between two training, 0 means no space. If no overlap type None. ')
            if epoch_overlap == 'None':
                epoch_overlap = None
            else:
                epoch_overlap = int(epoch_overlap)
            num_steps = int(input("Number of steps back in past of RNN you want to use : "))
            max_checks_without_progress = int(input('How many steps for no improvment for early stopping?: '))
            learning_rate = float(input('Learning rate? '))
            lambda_l2_reg = float(input('Penalty weight of L2 norm? '))
            num_epochs = float(input('Maximum number of epochs? ' ))
                
            if cv_method == 'IC':
                import IC

                if one_std:
                    print('BIC is recommended to prefer a simplier model for robustness.')
                    
                IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')
                if IC_method == 'None':
                    IC_method = None
                    
    
                print('------Model Construction------')
    
                RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= IC.IC_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = IC_method, cell_type = cell_type,\
                                                                                                                                        activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                        batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                        num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
                
                

            else:
                K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
                
                print('------Model Construction------')

                if not one_std:
                    
                    import cv_final as cv
        
                    RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= cv.CV_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, K_fold = K_fold, Nr= Nr, cell_type = cell_type,group=group,\
                                                                                                                                            activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                            batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                            num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
                else:
                    import cv_final_onestd as cv_std    
                    print('CV with ons-std rule is used for RNN model')

        
                    RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= cv_std.CV_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, K_fold = K_fold, Nr= Nr, cell_type = cell_type,group=group,\
                                                                                                                                            activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                            batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                            num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
               
                    
                    
            #k-step prediction
            num_train=round(X.shape[0]*RNN_hyper['early_stop']['train_ratio'])
            
            print('K-step prediction for training')
            train_y_prediction_kstep, train_loss_kstep = t_RNN.timeseries_RNN_feedback_test(X_scale[:num_train], y_scale[:num_train], X_scale[:num_train],y_scale[:num_train], kstep = steps, cell_type=RNN_hyper['cell_type'],activation = RNN_hyper['activation'], state_size = RNN_hyper['state_size'],\
                                                                                            num_layers = RNN_hyper['num_layers'], location=RNN_model, plot=True, round_number = str(round_number))
            
            print('K-step prediction for testing')
            test_y_prediction_kstep, test_loss_kstep = t_RNN.timeseries_RNN_feedback_test(X_scale[:num_train], y_scale[:num_train], X_test_scale,y_test_scale, kstep = steps, cell_type=RNN_hyper['cell_type'],activation = RNN_hyper['activation'], state_size = RNN_hyper['state_size'],\
                                                                                          num_layers = RNN_hyper['num_layers'], location=RNN_model, plot=True, round_number = str(round_number))
            scaler_y_RNN = StandardScaler(with_mean=True, with_std=True)
            scaler_y_RNN.fit(y_scale[:num_train])
            yhat_test_RNN = scaler_y_RNN.inverse_transform(yhat_test_RNN)
        
            residual_analysis(X_test_scale, y_test_scale, yhat_test_RNN, alpha = alpha, round_number = round_number)
                    
            
            fitting_result[selected_model] = {'model_hyper':RNN_hyper,'final_model':RNN_model, 'mse_train':mse_train_RNN, 'mse_val':mse_val_RNN, 'mse_test':mse_test_RNN, 'yhat_train':yhat_train_RNN, 'yhat_val':yhat_val_RNN, 'yhat_test':yhat_test_RNN, 'MSE_val': MSE_val}

            print('--------------Analysis Is Done--------------')

                    
                    
            


        else:
            import regression_models as rm
            
            alpha_num = int(input('Number of penalty weight you want to consider in DALVEN, if not known input 20: '))
            lag = list(map(int,input("Lists of numbers of lags you want to consider in DALVEN: (e.g. 1 2 3) ").strip().split()))
            degree = list(map(int,input("Orders of nonlinear mapping considered in DALVEN: (choose to include 1 2 3) ").strip().split()))

            if int(input('Do you want to test both DALVEN-full/DALVEN? (Yes: 1, No: 0): ')):
                if cv_method == 'IC':
                    import IC
                    
                    if one_std:
                        print('BIC is recommended to prefer a simplier model for robustness.')
                    
                    IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')

                    print('------Model Construction------')
                    
                    DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = IC.IC_mse('DALVEN', X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
                    DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = IC.IC_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
    
                else:
                    #using validation set
                    if not one_std:
                        import cv_final as cv
    
                        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                        print('------Model Construction------')

     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv.CV_mse('DALVEN', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                        DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = cv.CV_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)                       
                        
                    
                    else:
                        print('CV with ons-std rule is used for DALVEN model')
                        import cv_final_onestd as cv_std    
                        
                        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                        print('------Model Construction------')
     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv_std.CV_mse('DALVEN', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                        DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = cv_std.CV_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                
                ##select the method
                if MSE_v_DALVEN <= MSE_v_DALVEN_full:
                    selected_model = 'DALVEN'
                else:
                    selected_model = 'DALVEN_full_nonlinear'
                        
                print('Based on the ' + cv_method +', ' + selected_model +' is selected.')
                            
                                               
                #DALVEN model evaluation after choosing the method
                if selected_model == 'DALVEN':
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))

                    fitting_result[selected_model] = {'model_hyper':DALVEN_hyper,'final_model':DALVEN_model, 'model_params':DALVEN_params , 'mse_train':mse_train_DALVEN, 'mse_val':MSE_v_DALVEN, 'mse_test':mse_test_DALVEN, 'yhat_train':yhat_train_DALVEN, 'yhat_test':yhat_test_DALVEN, 'final_list': final_list}

                    lag_number = DALVEN_hyper['lag']
                                    
                    scalery = StandardScaler()
                    scalery.fit(y[lag_number:])
                                
                    yhat_test_DALVEN=scalery.inverse_transform(yhat_test_DALVEN)
                        
                                              
                    residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN, alpha =alpha, round_number = round_number)
            
                else:
                              
                    DALVEN_full_mse_test_multi, DALVEN_full_yhat_test_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X_test, y_test, DALVEN_full_model,DALVEN_full_hyper['retain_index'], DALVEN_full_hyper['degree'], DALVEN_full_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number = str(round_number))
                    DALVEN_full_mse_train_multi, DALVEN_full_yhat_train_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X, y, DALVEN_full_model,DALVEN_full_hyper['retain_index'], DALVEN_full_hyper['degree'], DALVEN_full_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number =str(round_number))
               
                    fitting_result[selected_model] = {'model_hyper':DALVEN_full_hyper,'final_model':DALVEN_full_model, 'model_params':DALVEN_full_params , 'mse_train':mse_train_DALVEN_full, 'mse_val':MSE_v_DALVEN_full, 'mse_test':mse_test_DALVEN_full, 'yhat_train':yhat_train_DALVEN_full, 'yhat_test':yhat_test_DALVEN_full, 'final_list': final_list_full}

                    lag_number = DALVEN_full_hyper['lag']
                                    
                    scalery = StandardScaler()
                    scalery.fit(y[lag_number:])
                                
                    yhat_test_DALVEN_full=scalery.inverse_transform(yhat_test_DALVEN_full)
                         
                                              
                    residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN_full, alpha =alpha, round_number = round_number)
            
                print('--------------Analysis Is Done--------------')
                    
                        

            
            
            
            else:
                DALVEN_method = input('Type in the method you want to test: DALVEN_full_nonlinear or DALVEN. ')
                selected_model = DALVEN_method
                
                if cv_method == 'IC':
                    import IC
                    
                    if one_std:
                        print('BIC is recommended to prefer a simplier model for robustness.')
                    
                    IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')

                    print('------Model Construction------')

                    DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = IC.IC_mse(DALVEN_method, X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
    
                    
                    
                else:
                    #using validation set
                    K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                    Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                    print('------Model Construction------')
                    
                    if not one_std:
                        import cv_final as cv
         
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv.CV_mse(DALVEN_method, X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                       
                                        
                    else:
                        print('CV with ons-std rule is used for DALVEN model')
                        import cv_final_onestd as cv_std    
                        
     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv_std.CV_mse(DALVEN_method, X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                
                
                fitting_result[selected_model] = {'model_hyper':DALVEN_hyper,'final_model':DALVEN_model, 'model_params':DALVEN_params , 'mse_train':mse_train_DALVEN, 'mse_val':MSE_v_DALVEN, 'mse_test':mse_test_DALVEN, 'yhat_train':yhat_train_DALVEN, 'yhat_test':yhat_test_DALVEN, 'final_list': final_list}

                ### model evaluation
                if DALVEN_method == 'DALVEN':
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                        
                else:                              
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number =str(round_number))
               
                lag_number = DALVEN_hyper['lag']
                                
                scalery = StandardScaler()
                scalery.fit(y[lag_number:])
                            
                yhat_test_DALVEN=scalery.inverse_transform(yhat_test_DALVEN)
                residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN, alpha =alpha, round_number = round_number)
        
                print('--------------Analysis Is Done--------------')
                    
                        

                
    
    else:
        import timeseries_regression_matlab as t_matlab
        
        maxorder = int(input('Maximum order number you want to consider: '))
        
        print('------Model Construction------')
        #matlab
        matlab_params, matlab_myresults, matlab_MSE_train, matlab_MSE_val, matlab_MSE_test, matlab_y_predict_train, matlab_y_predict_val, matlab_y_predict_test, matlab_train_error, matlab_val_error, matlab_test_error = t_matlab.timeseries_matlab_single(X, y, X_test = X_test, y_test= y_test, train_ratio = 1,\
                                                                                                                                                                                                                                                             maxorder = maxorder, mynow = 1, steps = steps, plot = True)
        #adaptx
        ADAPTx = int(input('Do you installed ADAPTx software? [Yes: 1, No: 0] '))
        if ADAPTx:
            import timeseries_regression_Adaptx as t_Adaptx
    
            url = input('Url for the ADAPTx software (e.g. C:\\Users\\Vicky\\Desktop\\AdaptX\\ADAPTX35M9\\): ')
            data_url =  input('Saved data file URL for the ADAPTx software (e.g. C:\\Users\\Vicky\\Desktop\\AdaptX\\ADAPTX35M9\\test\\): ')
            mymaxlag = int(input('Maximum number of lags considered in ADAPTx: '))
            mydegs = [int(x) for x in input('Degree of trend t in ADAPTx [if not known use default -1 0 1]: ').split()]
            
            
            Adaptx_optimal_params, Adaptx_myresults, Adaptx_MSE_train, Adaptx_MSE_val, Adaptx_MSE_test, Adaptx_y_predict_train, Adaptx_y_predict_val, Adaptx_y_predict_test, Adaptx_train_error, Adaptx_val_error, Adaptx_test_error = t_Adaptx.Adaptx_matlab_single(X, y, data_url = data_url, url = url, X_test=X_test, y_test=y_test, train_ratio = 1,\
                                                                                                                                                mymaxlag = mymaxlag, mydegs = mydegs, mynow = 1, steps = steps, plot = True)                     
                    
        
        if not ADAPTx: 
            print('The final state space model is fitted based on ' + matlab_params['method'][0] + ' using MATLAB.')
            selected_model = matlab_params['method'][0]
            yhat_test = matlab_y_predict_test.transpose()[:,0].reshape((-1,1))
            yhat_test = scaler_y.inverse_transform(yhat_test)
            
            fitting_result[selected_model] = {'model_hyper':matlab_params,'final_model':matlab_myresults, 'mse_train':matlab_MSE_train, 'mse_val':matlab_MSE_val, 'mse_test':matlab_MSE_test, 'yhat_train':matlab_y_predict_train, 'yhat_val':matlab_y_predict_val,'yhat_test':matlab_y_predict_test}

            residual_analysis(X_test, y_test,yhat_test, alpha = alpha, round_number = round_number)
            print('--------------Analysis Is Done--------------')
            
        else:
            print('The final state space model is selected based on minimum averaged MSE.')
            if np.mean(matlab_MSE_train) <= np.mean(Adaptx_MSE_train):
                print('State space model by Matlab is selected. Model parameters is stored in matlab_params.')
                print('The final state space model is fitted based on ' + matlab_params['method'][0] + ' using MATLAB.')
                selected_model = matlab_params['method'][0]
                
                fitting_result[selected_model] = {'model_hyper':matlab_params,'final_model':matlab_myresults, 'mse_train':matlab_MSE_train, 'mse_val':matlab_MSE_val, 'mse_test':matlab_MSE_test, 'yhat_train':matlab_y_predict_train, 'yhat_val':matlab_y_predict_val,'yhat_test':matlab_y_predict_test}
                
                yhat_test = matlab_y_predict_test.transpose()[:,0].reshape((-1,1))
                yhat_test = scaler_y.inverse_transform(yhat_test)

                residual_analysis(X_test, y_test,yhat_test, alpha = alpha, round_number = round_number)
                print('--------------Analysis Is Done--------------')
                
                
            else:
                print('State space model by ADAPTx is selected. Model parameters is stored in Adaptx_optimal_params.')
                selected_model = 'ADAPTx'
                fitting_result[selected_model] = {'model_hyper':Adaptx_optimal_params,'final_model':Adaptx_myresults, 'mse_train':Adaptx_MSE_train, 'mse_val':Adaptx_MSE_val, 'mse_test':Adaptx_MSE_test, 'yhat_train':Adaptx_y_predict_train, 'yhat_val':Adaptx_y_predict_val,'yhat_test':Adaptx_y_predict_test}

                yhat_test = Adaptx_y_predict_test.transpose()[Adaptx_optimal_params['lag'][0][0]:,0].reshape((-1,1))
                yhat_test = scaler_y.inverse_transform(yhat_test)                
                residual_analysis(X_test[Adaptx_optimal_params['lag'][0][0]:,:], y_test[Adaptx_optimal_params['lag'][0][0]:,0].reshape((-1,1)),yhat_test, alpha = alpha, round_number = round_number)
                print('--------------Analysis Is Done--------------')




############################ model fitting 2nd round if necessary
if round_number == 2:

    print('')
    print("""-----------------------------------------------------
The residual contains dynamics unexplained by the static model, therefore the dynamic model is selected.
-----------------------------------------""")
    
    print('')
    nonlinear_dynamic = int(input('Do you want to use a nonlinear dynamic model (Yes 1 No 0 Not sure 2): '))
    
    if nonlinear_dynamic == 1:
        if_nonlinear = 1
    elif nonlinear_dynamic ==2:
        if_nonlinear = nonlinearity_assess(X_original, y_original, if_plot_interrogation, cat = cat,alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number =  round_number)
        if if_nonlinear == 0:
            lag = int(input('The lag number you want to use to assess nonlinear dyanmics: '))
            if_nonlinear_dynamic = nonlinearity_assess_dynamic(X_original, y_original, if_plot_interrogation, alpha = alpha, difference = 0.4, xticks = xticks, yticks = yticks, round_number =  round_number,lag= lag)
            if_nonlinear = if_nonlinear or if_nonlinear_dynamic
    
    else:
        if_nonlinear == 0
        
        
        
    print('')
    print("""----------------------------------------------------
Based on the information of data characteristics, the following dynamic model is selected:
----------------------------------------------------""")
        
    model_name = None
            
    if if_nonlinear == 1:
        print('The nonlinear dynamic model is selected:')
        if if_enough == 0 :
            print('Because you have limited data, DALVEN is recommonded.')
            model_name = ['DALVEN']
        elif if_interpretable == 1:
            print('Because you would like an interpretable model, DALVEN is recommonded.')
            model_name = ['DALVEN']            
        else:
            print('Because you have engough data and do not require interpretability, RNN is recommonded.')
            model_name = ['RNN']
            
    else:
        print('There is significant dynamics and multicolinearity, CVA/SSARX/MOSEP are recommonded.')
        model_name = ['SS']
        
    
    print('')
    print("""----------------------------------------------------
Based on the information of data characteristics, the following fitting strategy is selected:
----------------------------------------------------""")
        
    if model_name == ['SS']:
        print('MATLAB/ADAPTx packges with information criterion will be used')
        
    else:    
        if if_enough == 1:
            cv_method = 'Single_ordered'
            print('Single validation is used for time series modeling.')
        else:
            if if_time:
                if if_robust:
                    cv_method = 'Timeseries'
                    one_std = 1
                    print('Cross-validation for time series with one std rule is selected.')
                                           
                else:
                    cv_method = 'Timeseries'
                    print('Cross-validation for time series is selected.')
            else:
                cv_method = 'IC'
                print('Information criteria is selected.')
                if if_robust:
                    one_std = 1
        
    print('')
    print("""----------------------------------------------------
Start 2nd-Round Model Fitting
----------------------------------------------------""")
    steps = int(input('Number of steps you want to test for the future prediction? '))
    
    if if_nonlinear == 1:

        if model_name == ['RNN']:
            selected_model = 'RNN'
            import timeseries_regression_RNN as t_RNN
            
            print('Please input the following numbers/types from the smallest to the largest: 1 3 5 or linear relu')
            activation = list(map(str,input("Types of activation function you want to use (e.g. linear, relu, tanh?): ").strip().split()))
            num_layers = list(map(int,input("Numbers of layers you want to test: ").strip().split()))
            state_size = list(map(int,input("Numbers of states you want to test: ").strip().split()))
            cell_type = list(map(str,input("Types of cells you want to use (e.g. regular, LSTM, GRU?): ").strip().split()))

            print('Please provide the following training parameter.')
            batch_size = int(input("Number of batch used in training you want to test: "))
            epoch_overlap = input('The overlap between different batch? The number indicate the space between two training, 0 means no space. If no overlap type None. ')
            if epoch_overlap == 'None':
                epoch_overlap = None
            else:
                epoch_overlap = int(epoch_overlap)
            num_steps = int(input("Number of steps back in past of RNN you want to use : "))
            max_checks_without_progress = int(input('How many steps for no improvment for early stopping?: '))
            learning_rate = float(input('Learning rate? '))
            lambda_l2_reg = float(input('Penalty weight of L2 norm? '))
            num_epochs = float(input('Maximum number of epochs? ' ))
                
            if cv_method == 'IC':
                import IC

                if one_std:
                    print('BIC is recommended to prefer a simplier model for robustness.')
                    
                IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')
                if IC_method == 'None':
                    IC_method = None
                    
    
                print('------Model Construction------')
    
                RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= IC.IC_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = IC_method, cell_type = cell_type,\
                                                                                                                                        activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                        batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                        num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
                
                

            else:
                K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     
                
                print('------Model Construction------')

                if not one_std:
                    
                    import cv_final as cv
        
                    RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= cv.CV_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, K_fold = K_fold, Nr= Nr, cell_type = cell_type,group=group,\
                                                                                                                                            activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                            batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                            num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
                else:
                    import cv_final_onestd as cv_std    
                    print('CV with ons-std rule is used for RNN model')

        
                    RNN_hyper, RNN_model, yhat_train_RNN, yhat_val_RNN, yhat_test_RNN, mse_train_RNN, mse_val_RNN, mse_test_RNN= cv_std.CV_mse('RNN', X_scale, y_scale, X_test_scale, y_test_scale, cv_type = cv_method, K_fold = K_fold, Nr= Nr, cell_type = cell_type,group=group,\
                                                                                                                                            activation = activation, num_layers=num_layers,state_size=state_size,num_steps=num_steps, \
                                                                                                                                            batch_size=batch_size,epoch_overlap= epoch_overlap, learning_rate=learning_rate, lambda_l2_reg=lambda_l2_reg,\
                                                                                                                                            num_epochs=num_epochs, max_checks_without_progress = max_checks_without_progress,round_number = str(round_number))
               
                    
                 
            fitting_result[selected_model] = {'model_hyper':RNN_hyper,'final_model':RNN_model, 'mse_train':mse_train_RNN, 'mse_val':mse_val_RNN, 'mse_test':mse_test_RNN, 'yhat_train':yhat_train_RNN, 'yhat_val':yhat_val_RNN, 'yhat_test':yhat_test_RNN, 'MSE_val': MSE_val}

            #k-step prediction
            num_train=round(X.shape[0]*RNN_hyper['early_stop']['train_ratio'])
            
            print('K-step prediction for training')
            train_y_prediction_kstep, train_loss_kstep = t_RNN.timeseries_RNN_feedback_test(X_scale[:num_train], y_scale[:num_train], X_scale[:num_train],y_scale[:num_train], kstep = steps, cell_type=RNN_hyper['cell_type'],activation = RNN_hyper['activation'], state_size = RNN_hyper['state_size'],\
                                                                                                                                      num_layers = RNN_hyper['num_layers'], location=RNN_model, plot=True,round_number = str(round_number))
            print('K-step prediction for training')
            test_y_prediction_kstep, test_loss_kstep = t_RNN.timeseries_RNN_feedback_test(X_scale[:num_train], y_scale[:num_train], X_test_scale,y_test_scale, kstep = steps, cell_type=RNN_hyper['cell_type'],activation = RNN_hyper['activation'], state_size = RNN_hyper['state_size'],\
                                                                                                                                      num_layers = RNN_hyper['num_layers'], location=RNN_model, plot=True, round_number = str(round_number))
            scaler_y_RNN = StandardScaler(with_mean=True, with_std=True)
            scaler_y_RNN.fit(y_scale[:num_train])
            yhat_test_RNN = scaler_y_RNN.inverse_transform(yhat_test_RNN)
        
            residual_analysis(X_test_scale, y_test_scale, yhat_test_RNN, alpha = alpha, round_number = round_number)
                    
            print('--------------Analysis Is Done--------------')

                    
                    
            


        else:
            import regression_models as rm

            alpha_num = int(input('Number of penalty weight you want to consider in DALVEN, if not known input 20: '))
            lag = list(map(int,input("Lists of numbers of lags you want to consider in DALVEN: (e.g. 1 2 3) ").strip().split()))
            degree = list(map(int,input("Orders of nonlinear mapping considered in DALVEN: (choose to include 1 2 3) ").strip().split()))

            if int(input('Do you want to test both DALVEN-full/DALVEN? (Yes: 1, No: 0): ')):
                if cv_method == 'IC':
                    import IC
                    
                    if one_std:
                        print('BIC is recommended to prefer a simplier model for robustness.')
                    
                    IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')

                    print('------Model Construction------')
                    
                    DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = IC.IC_mse('DALVEN', X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
                    DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = IC.IC_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
    
                else:
                    #using validation set
                    if not one_std:
                        import cv_final as cv
    
                        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                        print('------Model Construction------')

     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv.CV_mse('DALVEN', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                        DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = cv.CV_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)                       
                        
                    
                    else:
                        print('CV with ons-std rule is used for DALVEN model')
                        import cv_final_onestd as cv_std    
                        
                        K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                        Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                        print('------Model Construction------')
     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv_std.CV_mse('DALVEN', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                        DALVEN_full_hyper,DALVEN_full_model, DALVEN_full_params, mse_train_DALVEN_full, mse_test_DALVEN_full, yhat_train_DALVEN_full, yhat_test_DALVEN_full, MSE_v_DALVEN_full, final_list_full = cv_std.CV_mse('DALVEN_full_nonlinear', X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                
                ##select the method
                if MSE_v_DALVEN <= MSE_v_DALVEN_full:
                    selected_model = 'DALVEN'
                else:
                    selected_model = 'DALVEN_full_nonlinear'
                        
                print('Based on the ' + cv_method +', ' + selected_model +' is selected.')
                            
                                               
                #DALVEN model evaluation after choosing the method
                if selected_model == 'DALVEN':
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))

                    lag_number = DALVEN_hyper['lag']
                                    
                    scalery = StandardScaler()
                    scalery.fit(y[lag_number:])
                                
                    yhat_test_DALVEN=scalery.inverse_transform(yhat_test_DALVEN)
                         
                    fitting_result[selected_model] = {'model_hyper':DALVEN_hyper,'final_model':DALVEN_model, 'model_params':DALVEN_params , 'mse_train':mse_train_DALVEN, 'mse_val':MSE_v_DALVEN, 'mse_test':mse_test_DALVEN, 'yhat_train':yhat_train_DALVEN, 'yhat_test':yhat_test_DALVEN, 'final_list': final_list}
                                              
                    residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN, alpha =alpha, round_number = round_number)
                                  
                else:
                              
                    DALVEN_full_mse_test_multi, DALVEN_full_yhat_test_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X_test, y_test, DALVEN_full_model,DALVEN_full_hyper['retain_index'], DALVEN_full_hyper['degree'], DALVEN_full_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number = str(round_number))
                    DALVEN_full_mse_train_multi, DALVEN_full_yhat_train_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X, y, DALVEN_full_model,DALVEN_full_hyper['retain_index'], DALVEN_full_hyper['degree'], DALVEN_full_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number =str(round_number))
               
                    lag_number = DALVEN_full_hyper['lag']
                                    
                    scalery = StandardScaler()
                    scalery.fit(y[lag_number:])
                                
                    yhat_test_DALVEN_full=scalery.inverse_transform(yhat_test_DALVEN_full)
                         
                    fitting_result[selected_model] = {'model_hyper':DALVEN_full_hyper,'final_model':DALVEN_full_model, 'model_params':DALVEN_full_params , 'mse_train':mse_train_DALVEN_full, 'mse_val':MSE_v_DALVEN_full, 'mse_test':mse_test_DALVEN_full, 'yhat_train':yhat_train_DALVEN_full, 'yhat_test':yhat_test_DALVEN_full, 'final_list': final_list_full}
                           
                    residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN_full, alpha =alpha, round_number = round_number)
            
                print('--------------Analysis Is Done--------------')
                    
                        

            
            
            
            else:
                DALVEN_method = input('Type in the method you want to test: DALVEN_full_nonlinear or DALVEN. ')
                selected_model = DALVEN_method
                
                if cv_method == 'IC':
                    import IC
                    
                    if one_std:
                        print('BIC is recommended to prefer a simplier model for robustness.')
                    
                    IC_method = input('The type of information criterion you want to use (AIC, AICc, BIC), if not known type None, the criterion will be selected between AIC/AICc. ')

                    print('------Model Construction------')

                    DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = IC.IC_mse(DALVEN_method, X, y, X_test, y_test, cv_type = IC_method, alpha_num=alpha_num, lag=lag, degree=degree, label_name=True, trans_type= 'auto')
    
                    
                    
                else:
                    #using validation set
                    K_fold = int(input('Number of K-fold you want to use, or the fold number you want to use in single validation 1/K, if not known input 5: '))
                    Nr = int(input('Number of repetition (if have in CV) you want to use, if not known input 10: '))     

                    print('------Model Construction------')
                    
                    if not one_std:
                        import cv_final as cv
         
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv.CV_mse(DALVEN_method, X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                       
                                        
                    else:
                        print('CV with ons-std rule is used for DALVEN model')
                        import cv_final_onestd as cv_std    
                        
     
                        DALVEN_hyper,DALVEN_model, DALVEN_params, mse_train_DALVEN, mse_test_DALVEN, yhat_train_DALVEN, yhat_test_DALVEN, MSE_v_DALVEN, final_list = cv_std.CV_mse(DALVEN_method, X, y, X_test, y_test, cv_type = cv_method, K_fold = K_fold, Nr= Nr, \
                                                                                                                                                                               alpha_num=alpha_num, label_name=True, trans_type= 'auto',degree=degree,lag = lag)
                
                fitting_result[selected_model] = {'model_hyper':DALVEN_hyper,'final_model':DALVEN_model, 'model_params':DALVEN_params , 'mse_train':mse_train_DALVEN, 'mse_val':MSE_v_DALVEN, 'mse_test':mse_test_DALVEN, 'yhat_train':yhat_train_DALVEN, 'yhat_test':yhat_test_DALVEN, 'final_list': final_list}

                ### model evaluation
                if DALVEN_method == 'DALVEN':
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps,trans_type = 'auto',plot=True,round_number = str(round_number))
                        
                else:                              
                    DALVEN_mse_test_multi, DALVEN_yhat_test_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X_test, y_test, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number = str(round_number))
                    DALVEN_mse_train_multi, DALVEN_yhat_train_multi= rm.DALVEN_testing_kstep_full_nonlinear(X, y, X, y, DALVEN_model,DALVEN_hyper['retain_index'], DALVEN_hyper['degree'], DALVEN_hyper['lag'] , steps ,trans_type = 'auto', plot=True,round_number =str(round_number))
               
                lag_number = DALVEN_hyper['lag']
                                
                scalery = StandardScaler()
                scalery.fit(y[lag_number:])
                            
                yhat_test_DALVEN=scalery.inverse_transform(yhat_test_DALVEN)
                residual_analysis(X_test[lag_number:], y_test[lag_number:],yhat_test_DALVEN, alpha =alpha, round_number = round_number)
        
                print('--------------Analysis Is Done--------------')
                    
                        

                
    
    else:
        import timeseries_regression_matlab as t_matlab
        
        maxorder = int(input('Maximum order number you want to consider: '))
        
        print('------Model Construction------')
        #matlab
        matlab_params, matlab_myresults, matlab_MSE_train, matlab_MSE_val, matlab_MSE_test, matlab_y_predict_train, matlab_y_predict_val, matlab_y_predict_test, matlab_train_error, matlab_val_error, matlab_test_error = t_matlab.timeseries_matlab_single(X, y, X_test = X_test, y_test= y_test, train_ratio = 1,\
                                                                                                                                                                                                                                                             maxorder = maxorder, mynow = 1, steps = steps, plot = True)
        #adaptx
        ADAPTx = int(input('Do you installed ADAPTx software? [Yes: 1, No: 0] '))
        if ADAPTx:
            import timeseries_regression_Adaptx as t_Adaptx
    
            url = input('Url for the ADAPTx software (e.g. C:\\Users\\Vicky\\Desktop\\AdaptX\\ADAPTX35M9\\): ')
            data_url =  input('Saved data file URL for the ADAPTx software (e.g. C:\\Users\\Vicky\\Desktop\\AdaptX\\ADAPTX35M9\\test\\): ')
            mymaxlag = int(input('Maximum number of lags considered in ADAPTx: '))
            mydegs = [int(x) for x in input('Degree of trend t in ADAPTx [if not known use default -1 0 1]: ').split()]
            
            
            Adaptx_optimal_params, Adaptx_myresults, Adaptx_MSE_train, Adaptx_MSE_val, Adaptx_MSE_test, Adaptx_y_predict_train, Adaptx_y_predict_val, Adaptx_y_predict_test, Adaptx_train_error, Adaptx_val_error, Adaptx_test_error = t_Adaptx.Adaptx_matlab_single(X, y, data_url = data_url, url = url, X_test=X_test, y_test=y_test, train_ratio = 1,\
                                                                                                                                                mymaxlag = mymaxlag, mydegs = mydegs, mynow = 1, steps = steps, plot = True)                     
                    
        
        if not ADAPTx: 
            print('The final state space model is fitted based on ' + matlab_params['method'][0] + ' using MATLAB.')
            selected_model = matlab_params['method'][0]
            yhat_test = matlab_y_predict_test.transpose()[:,0].reshape((-1,1))
            yhat_test = scaler_y.inverse_transform(yhat_test)
            
            fitting_result[selected_model] = {'model_hyper':matlab_params,'final_model':matlab_myresults, 'mse_train':matlab_MSE_train, 'mse_val':matlab_MSE_val, 'mse_test':matlab_MSE_test, 'yhat_train':matlab_y_predict_train, 'yhat_val':matlab_y_predict_val,'yhat_test':matlab_y_predict_test}

            residual_analysis(X_test, y_test,yhat_test, alpha = alpha, round_number = round_number)
            print('--------------Analysis Is Done--------------')
            
        else:
            print('The final state space model is selected based on minimum averaged MSE.')
            if np.mean(matlab_MSE_train) <= np.mean(Adaptx_MSE_train):
                print('State space model by Matlab is selected. Model parameters is stored in matlab_params.')
                print('The final state space model is fitted based on ' + matlab_params['method'][0] + ' using MATLAB.')
                selected_model = matlab_params['method'][0]
                yhat_test = matlab_y_predict_test.transpose()[:,0].reshape((-1,1))
                yhat_test = scaler_y.inverse_transform(yhat_test)
                fitting_result[selected_model] = {'model_hyper':matlab_params,'final_model':matlab_myresults, 'mse_train':matlab_MSE_train, 'mse_val':matlab_MSE_val, 'mse_test':matlab_MSE_test, 'yhat_train':matlab_y_predict_train, 'yhat_val':matlab_y_predict_val,'yhat_test':matlab_y_predict_test}

                residual_analysis(X_test, y_test,yhat_test, alpha = alpha, round_number = round_number)
                print('--------------Analysis Is Done--------------')
                
                
            else:
                print('State space model by ADAPTx is selected. Model parameters is stored in Adaptx_optimal_params.')
                selected_model = 'ADAPTx'
                yhat_test = Adaptx_y_predict_test.transpose()[Adaptx_optimal_params['lag'][0][0]:,0].reshape((-1,1))
                yhat_test = scaler_y.inverse_transform(yhat_test)                
                fitting_result[selected_model] = {'model_hyper':Adaptx_optimal_params,'final_model':Adaptx_myresults, 'mse_train':Adaptx_MSE_train, 'mse_val':Adaptx_MSE_val, 'mse_test':Adaptx_MSE_test, 'yhat_train':Adaptx_y_predict_train, 'yhat_val':Adaptx_y_predict_val,'yhat_test':Adaptx_y_predict_test}

                residual_analysis(X_test[Adaptx_optimal_params['lag'][0][0]:,:], y_test[Adaptx_optimal_params['lag'][0][0]:,0].reshape((-1,1)),yhat_test, alpha = alpha, round_number = round_number)
                print('--------------Analysis Is Done--------------')