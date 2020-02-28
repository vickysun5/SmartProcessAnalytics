%data, dimy, mymaxlag mydegs
clear all
clc
close all

load myfilist.mat   %training dataset, saved in  DATAIN AND DATAFC (optional)
load myparams.mat   %parameter set, saved in MYPARAMS


if exist('mydataval.mat')==2
    load mydataval.mat
else
    mydataval = [];
end


current_url = pwd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set up path
% url = 'C:\Users\Vicky\Desktop\AdaptX\ADAPTX35M9\';

path([url 'exam'],path)
path([url 'dotm'],path)
path([url 'pcode'],path)
path(url,path)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%clean old files except the data file
DataPath = data_url;
f=dir([DataPath  '*.mat']);
f={f.name};
for i = 1:num_series
    n = find(strcmp(f, ['filist' num2str(i) '.mat']));
    f{n} = '';
end

for k = 1:numel(f)
    if strcmp(f{k},'') || strcmp(f{k},'.') || strcmp(f{k},'..')
    else
       delete([DataPath f{k}])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% setting parameters and run
mydimy = double(mydimy);
mydimu = double(mydimu);
mymaxlag = double(mymaxlag);
mydegs=double(mydegs);
mynow = double(mynow);
steps= double(steps);
timax = double(timax);
val=double(val);  %flag for validation data or not

vicky_auto_multiple_program(url, data_url, filist, timax, mydataval, mydimy, mydimu, mymaxlag, mydegs, mynow, steps, val);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% reutrn back to the main folder for
%%%%%%%%%%clc%%%%%%%%%%%%%%%%%%%%%%%%%% Smart Data Analytics
cd(current_url)







function [] = vicky_auto_multiple_program(url, data_url, filist, timax, mydataval, mydimy, mydimu, mymaxlag, mydegs, mynow, steps, val)
% sets up file mvdir.mat with string variable mvname as "adaptx" directory
mvname = url;
eval(['save ' mvname '\mvdir mvname']);
eval(['cd ' data_url]);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%load data%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = [];
n = [];
save datain d n filist timax

dimy = mydimy;
dimu= mydimu;
maxlag = mymaxlag;
degs = mydegs;
now = mynow;
save lagin dimy dimu maxlag degs now

dfc=[];
nfc=[];
save datafc dfc nfc

confor =0.05;
save forcin steps confor


makeprn = 3;
save makeprn makeprn


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%run program%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%initial set up -- check data

if (exist('datain.mat')|exist('lagin.mat')) == 0
     disp('******************************************************')
     disp('Input data files DATAIN.MAT and/or LAGIN.MAT')
     disp('do not exist.  Exit the MENU and define them.')
     disp('******************************************************')
     return
end
if (exist('forcin.mat')|exist('datafc.mat')) == 0
     disp('****************************************************')
     disp('Files DATAFC.MAT and/or FORCIN.MAT needed in')
     disp('forecasting have not been defined.  If you')
     disp('want to forecast, exit the MENU and define them.')
     disp('****************************************************')
end
if (exist('confin.mat')|exist('specin.mat')|exist('timein.mat')) == 0
     disp('**********************************************************')
     disp('Plot parameter values have not been defined properly.')
     disp('Default plot parameters have been assigned.')
     disp('These may be changed in the CHANGE VALUES')
     disp('submenu of PLOT IDENTIFIED MODEL menu.')
     disp('**********************************************************')
     simref = 0;
     pts = 100;
     alpha = 0.05;
     save specin simref pts 
     save timein simref pts 
     save confin simref pts alpha 
end
% load  datain d n
% save data d n
% clear d

%%%%%%%%%%%%%%% ARX fit to find lag and deg orders
load  lagin dimy dimu maxlag degs now dobb arpreb arpree stpreb stpree
if exist('dobb') == 0, dobb = []; end,
if exist('arpreb') == 0, arpreb = []; end,
if exist('arpree') == 0, arpree = []; end,
if exist('stpreb') == 0, stpreb = []; end,
if exist('stpree') == 0, stpree = []; end,
save lag dimy dimu maxlag degs now dobb arpreb arpree stpreb stpree
disp('***********************************************')
disp('ARX models possibly including polynomials')
disp('in time are being fitted by ARLAG.M')
disp('***********************************************')
out = arlag(1);                             % Fit ARX models
load  arxcomp ar br abr bbr cr lagp lagf ardeg arord araic % Load ARX


if arord(ardeg+2)==0, disp('ARX Order = 0'), return, end, % No past dependence


%%%%%%%%%%%%CVA fit to find state order
disp('***********************************************')
disp('CVA of the process is being computed to ')
disp('determine the system states by CVAIC.M')
disp('***********************************************')
out = cvaic(1);                             % CVA and SS AICs
load  cvaic optord aici out                   % Load AICs & optimal SS order
ord = optord;
save myss ord                             % Save optimal SS order

%%%%%%%%%%%%%%%%fit SS model
disp('***********************************************')
disp('A state space model of the system is')
disp('being computed by SSMOD.M')
disp('***********************************************')
out = ssmod(1);                             % Fit SS model
disp('***********************************************')
disp('Confidence bands of the identified model')
disp('accuracy is being computed by CONFINT.M')
disp('***********************************************')
out = confint(1);                          % Compute confidence intervals


%%%%%%%%%%%%%%%%fit other model form
disp('***********************************************')
disp('An innovations state space model is')
disp('computed by INNOV.M')
disp('***********************************************')
out = innov(1);
disp('***********************************************')
disp('An echelon (overlapping parameterization)')
disp('model form is being computed by ECHELON.M')
disp('***********************************************')
out = echelon(0);
disp('***********************************************')
disp('An ARMA of ARMAX model form is being')
disp('computed by ARMA.M')
disp('***********************************************')
out = arma(0);



%% %%%%%%%%%%%%%load results%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ARX results
load arxcomp araic
load myopt mylag mydeg  %optimal lag
%CVA results
load cvaic aici
load myss ord   %optimal CVA state order
%SS results
load state phi g h a b q r e f abr bbr jk

% %other form
% load innov
% load echelon
% load arma



%save results in matfile
save myresults araic mylag mydeg aici ord phi g h a b q r e f abr bbr jk 





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%test on test dataset%%%%%
if val
    %delete old kstep files, do that for validation data
    clear ym yp ye merr
    delete([data_url , 'kstep.mat'])
    delete([data_url , 'predkal.mat'])
    
%   load mydataval.mat
    dval = mydataval;
    nval = size(dval,2);
    save dataval dval nval
    out = kstep(steps);
    load kstep ym yp ye merr
    save kstep_testing ym yp ye merr
    delete([data_url, 'kstep.mat'])
    delete([data_url, 'dataval.mat'])


end  

end


