clear all
clc
close all

load myparams.mat   %dimy, ntrain


%% load params
dimy = double(mydimy);
mynow = double(mynow);
steps= double(steps);
maxorder = double(maxorder);
test = double(test);
num_series = double(num_series);
train_ratio = double(train_ratio);



%% arrange the training data
%load the first one
load filist1.mat
y = d(1:dimy,:)';
u = d(dimy+1:end,:)';
dat = iddata(y,u,1);
clear d

for i = 2:num_series
    load(['filist', num2str(i), '.mat'])
    y = d(1:dimy,:)';
    u = d(dimy+1:end,:)';
    dat = merge(dat, iddata(y,u,1));
    clear d   
end

if train_ratio < 1
    load filist_whole1.mat
    y = d(1:dimy,:)';
    u = d(dimy+1:end,:)';
    dat_whole = iddata(y,u,1);
    ntrain = round(size(y,1)*train_ratio);
    dat_whole_val = iddata(y(ntrain+1:end,:),u(ntrain+1:end,:),1);
    clear d    

    for i = 2:num_series
        load(['filist_whole', num2str(i), '.mat'])
        y = d(1:dimy,:)';
        u = d(dimy+1:end,:)';
        dat_whole = merge(dat_whole, iddata(y,u,1));
        ntrain = round(size(y,1)*train_ratio);
        dat_whole_val = merge(dat_whole_val, iddata(y(ntrain+1:end,:),u(ntrain+1:end,:),1));
        clear d   
    end
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
        %get AIC and AICc
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



%% for train and validation
if train_ratio == 1
    for i = 1:num_series
        prediction = [];
        error = [];
        dat_single = getexp(dat,i);
        MSE = zeros(steps,size(dat_single.y,2));

        for k=1:steps
           yp = predict(sys,dat_single,k);
           prediction = [prediction,yp.y];
           error = [error,yp.y - dat_single.y];
           error_k=yp.y - dat.y;
           for j=1:size(MSE,2)
               MSE(k,j) = sum(error_k(k+1:end,j).^2)./size(error_k(k+1:end,j),1);
           end
        end
        
        prediction = prediction';
        error=error';
        MSE=MSE';
        save(['myresults_train', num2str(i), '.mat'],'prediction','error','MSE')
    end
    
else
    for i = 1:num_series
        dat_single = getexp(dat_whole,i);

        prediction = [];
        error = [];
        MSE = zeros(steps,size(dat_single.y,2));
        dat_train = getexp(dat,i);

        %% Calculate for train and validation data
        prediction_val = [];
        error_val = [];
        MSE_val = zeros(steps,size(dat_single.y,2));
        dat_val =getexp(dat_whole_val,i);
        
        ntrain = round(size(dat_single.y,1)*train_ratio);
        
        for k=1:steps
           yp = predict(sys,dat_single,k);

           yp_train = yp(1:ntrain,:);
           yp_val = yp(ntrain+1:end,:);
           
           prediction = [prediction,yp_train.y];
           error = [error,yp_train.y - dat_train.y];
           error_k=yp_train.y - dat_train.y;
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
        
        prediction = prediction';
        error=error';
        MSE=MSE';
        save(['myresults_train', num2str(i), '.mat'],'prediction','error','MSE')

        prediction_val = prediction_val';
        error_val = error_val';
        MSE_val = MSE_val';
        save(['myresults_val', num2str(i), '.mat'],'prediction_val','error_val','MSE_val')

    end
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

save my_matlab_ss sys A B C D K final_model final_order final_AIC
