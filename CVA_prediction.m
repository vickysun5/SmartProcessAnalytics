%data, dimy, mymaxlag mydegs
clear all
clc
close all

load myparams_prediction.mat   %parameter set, saved in MYPARAMS
load mydataval_prediction.mat


current_url = pwd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set up path
path([url 'exam'],path)
path([url 'dotm'],path)
path([url 'pcode'],path)
path(url,path)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% setting parameters and run
steps= double(steps);
id = double(id);

vicky_auto_prediction(url, data_url, mydataval_prediction,steps,id);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% reutrn back to the main folder
cd(current_url)


function [] = vicky_auto_prediction(url, data_url, mydataval_prediction, steps, id)
% sets up file mvdir.mat with string variable mvname as "adaptx" directory
mvname = url;
eval(['save ' mvname '\mvdir mvname']);
eval(['cd ' data_url]);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%check the existence of file%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (exist('innov.mat')) == 0
     disp('******************************************************')
     disp('Innovation Form Is Missing, Please Construct the Model First')
     disp('******************************************************')
     return
end

%%%%%%%%%%%%%%%%prediction for the validation set
delete([data_url , 'kstep.mat'])
delete([data_url , 'predkal.mat'])
    
%load mydataval.mat
dval = mydataval_prediction;
nval = size(dval,2);
save dataval dval nval
out = kstep(steps);
load kstep ym yp ye merr
save([data_url, 'kstep' , num2str(id), '.mat'], 'ym', 'yp', 'ye', 'merr')
delete kstep.mat
delete dataval.mat


end