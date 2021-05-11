function [TestingAccuracy,TestingAccuracyNC,TestingAccuracyAD] = HK_helm_train_1h(train_x,train_y,test_x,test_y,b1, Regularization_coefficient, Kernel_type, Kernel_para)
tic
train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];
clear train_x;
%% First layer RELM
A1 = H1 * b1;A1 = mapminmax(A1);
clear b1;
beta1  =  sparse_elm_autoencoder(A1,H1,1e-3,50)';
clear A1;

T1 = H1 * beta1;
% fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));

[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';

clear H1;

%% zyd
%%%%%%%%%%% Load training dataset
P=T1';
T=train_y';
clear train_y;

C = Regularization_coefficient;

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(T,2);
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T'));
 

%% First layer feedforward
tic;

test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
clear test_x;

TT1 = HH1 * beta1;TT1  =  mapminmax('apply',TT1',ps1)';
clear HH1;clear beta1;

%% zyd
%%%%%%%%%%% Load testing dataset
TV.T=test_y';%读取测试集标签1*样本数
TV.P=TT1';%读取测试集特征7616*样本数
clear test_y;

%%%%%%%%%%% Calculate the output of testing input
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data

%%%%%%%%%% Calculate training & testing classification accuracy
MissClassificationRate_Testing=0;
NC=0;
ADHDR=0;
NCR=0;

for i = 1 : size(TV.T, 2)
    [x, label_index_expected]=max(TV.T(:,i));
    [x, label_index_actual]=max(TY(:,i));
    if label_index_expected==1
        NC=NC+1;
        if label_index_actual==label_index_expected
            NCR=NCR+1;
        end
    else if label_index_actual==label_index_expected
            ADHDR=ADHDR+1;
        end
    end
    if label_index_actual~=label_index_expected
        MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  
TestingAccuracyNC=NCR/NC
TestingAccuracyAD=ADHDR/(size(TV.T,2)-NC)




%%
%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end



