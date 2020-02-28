clear all
clc
close all


load mydata.mat   % format save as for Adaptx, d x N
load myparams.mat   %dimy, ntrain

%% load params
dimy = double(mydimy);
mynow = double(mynow);
n_train = double(n_train);
steps= double(steps);
maxorder = double(maxorder);
test = double(test);



%% change the format
y = mydata(1:dimy,:)';
u = mydata(dimy+1:end,:)';

%% save in iddata
dat = iddata(y(1:n_train,:),u(1:n_train,:),1);
if n_train < size(y,1)
    dat_whole = iddata(y,u,1);
    dat_val = iddata(y(n_train+1:end,:),u(n_train+1:end,:),1);
end

%% save in iddata for test, if exist
if test
    load mydatatest.mat
    y_test = mydatatest(1:dimy,:)';
    u_test = mydatatest(dimy+1:end,:)';
    dat_test = iddata(y_test,u_test);
end



%% train
myoption = {'CVA','SSARX','MOESP'};
MSE = {[],[],[]};
AIC_min = [];
order_opt = [];


for index = 1:size(myoption,2)
    opt = n4sidOptions('N4weight',myoption{index});

    %% Esimated the ss model
    AIC=zeros(maxorder,1);
    AICc=zeros(maxorder,1);
    for i=1:size(AIC,1)
        sys = n4sid(dat,i, opt,'Feedthrough',mynow);
        %get AIC or AICc
        AIC(i)=sys.Report.Fit.AIC;
        AICc(i)=sys.Report.Fit.AICc;
    end
    
    K = sum(sys.Report.Parameters.Free);
    if n_train/K<40
        [aic,order] = min(AICc);
    else
        [aic,order] = min(AIC);
    end
    AIC_min = [AIC_min, aic];
    order_opt = [order_opt, order];
end

[final_AIC, final_index] = min(AIC_min);
final_model = myoption{final_index};
final_order = order_opt(final_index);

opt = n4sidOptions('N4weight',myoption{final_index});
sys = n4sid(dat,final_order, opt,'Feedthrough',mynow);


%% K-step prediction for training data and val (if needed)
prediction = [];
error = [];
MSE = zeros(steps,size(dat.y,2));

if n_train == size(y,1)
    for k=1:steps
       yp = predict(sys,dat,k);
       prediction = [prediction,yp.y];
       error = [error,yp.y - dat.y];
       error_k=yp.y - dat.y;
       for j=1:size(MSE,2)
           MSE(k,j) = sum(error_k(k+1:end,j).^2)./size(error_k(k+1:end,j),1);
       end
    end

else  
    %% Calculate for train and validation data
    prediction_val = [];
    error_val = [];
    MSE_val = zeros(steps,size(dat_val.y,2));

    for k=1:steps
       yp = predict(sys,dat_whole,k);
       
       yp_train = yp(1:n_train,:);
       yp_val = yp(n_train+1:end,:);
       
       prediction = [prediction,yp_train.y];
       error = [error,yp_train.y - dat.y];
       error_k=yp_train.y - dat.y;
       for j=1:size(MSE,2)
           MSE(k,j) = sum(error_k(k+1:end,j).^2)./size(error_k(k+1:end,j),1);
       end
       
       prediction_val = [prediction_val,yp_val.y];
       error_val = [error_val,yp_val.y - dat_val.y];
       error_k=yp_val.y - dat_val.y;
       for j=1:size(MSE_val,2)
            MSE_val(k,j) = sum(error_k(:,j).^2)./size(error_k(:,j),1);
       end
    end
    
    prediction_val = prediction_val';
    error_val = error_val';
    MSE_val = MSE_val';
    save myresults_val prediction_val error_val MSE_val
end

%% for test data
if test
    prediction_test = [];
    error_test = [];
    MSE_test = zeros(steps,size(dat_test.y,2)); 

    
    for k=1:steps
       yp = predict(sys,dat_test,k);
       prediction_test = [prediction_test,yp.y];
       error_test = [error_test,yp.y - dat_test.y];
       error_k=yp.y - dat_test.y;
       for j=1:size(MSE_test,2)
            MSE_test(k,j) = sum(error_k(k+1:end,j).^2)./size(error_k(k+1:end,j),1);
       end
    end
    
    prediction_test = prediction_test';
    error_test = error_test';
    MSE_test = MSE_test';
    
    save myresults_test prediction_test error_test MSE_test
end

%save results in matfile
A=sys.A;
B=sys.B;
C=sys.C;
D=sys.D;
K=sys.K;

prediction = prediction';
error=error';
MSE=MSE';
save myresults_train prediction error MSE
save my_matlab_ss sys A B C D K final_model final_order final_AIC