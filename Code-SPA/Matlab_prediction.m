load myparams_prediction.mat
load mydata_prediction.mat

%% load params
steps= double(steps);
id = double(id);
dimy = double(mydimy);


%% change the format
y = mydata_prediction(1:dimy,:)';
u = mydata_prediction(dimy+1:end,:)';

%% save in iddata
dat = iddata(y,u,1);


%% load the system
load(mysys+".mat");

%% prediction
prediction = [];
error = [];
MSE = zeros(steps,size(dat.y,2));

for k=1:steps
   yp = predict(sys,dat,k);
   prediction = [prediction,yp.y];
   error = [error,yp.y - dat.y];
   error_k=yp.y - dat.y;
   for j=1:size(MSE,2)
       MSE(k,j) = sum(error_k(k+1:end,j).^2)./size(error_k(k+1:end,j),1);
   end
end

prediction = prediction';
error = error';
MSE=MSE';

save(['kstep' , num2str(id), '.mat'], 'prediction','error','MSE')
