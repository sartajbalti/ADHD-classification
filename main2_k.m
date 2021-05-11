%clear;
clear abnormal normal train_label test_label;
load Numof5Fold.mat;
count=0;
%%%data
for i=1:5
    %read normal
    eval(['maindir_nc = ''.\flod5data\flod',num2str(i),'\normal'';']);
    dirpath_nc=fullfile( maindir_nc, '*.txt' );
    brains_nc = dir(dirpath_nc );
    for j = 1:normal_num(i)
         brainpath = fullfile( maindir_nc, brains_nc(j).name  );
         I = load( brainpath );   % 这里进行你的读取操作 358*t
         O=LocalFeatureExtract13(I);%64*119
         eval(['part',num2str(i),'_normal(:,:,j)=O;']);
         count=count+1;
         flodindex(count,1)=i;
    end
    %label
    eval(['part',num2str(i),'_normal_label=zeros(normal_num(',num2str(i),'),1);']);
    eval(['tmp =reshape(part',num2str(i),'_normal,7616,normal_num(',num2str(i),'));']);
    eval(['part',num2str(i),'_normal=tmp'';']);%127*7616
    %read abnormal
    eval(['maindir_ad = ''.\flod5data\flod',num2str(i),'\abnormal'';']);
    dirpath_ad=fullfile( maindir_ad, '*.txt' );
    brains_ad = dir(dirpath_ad );
    for j = 1:abnormal_num(i)
         brainpath = fullfile( maindir_ad, brains_ad(j).name  );
         I = load( brainpath );   % 这里进行你的读取操作 358*t
         O=LocalFeatureExtract13(I);%64*119
         eval(['part',num2str(i),'_abnormal(:,:,j)=O;']);
         count=count+1;
         flodindex(count,1)=i;
    end
    eval(['part',num2str(i),'_abnormal_label=ones(abnormal_num(',num2str(i),'),1);']);
    eval(['tmp =reshape(part',num2str(i),'_abnormal,7616,abnormal_num(',num2str(i),'));']);
    eval(['part',num2str(i),'_abnormal=tmp'';']);%127*7616
end
%%
data=[part1_normal;part1_abnormal;
    part2_normal;part2_abnormal;
    part3_normal;part3_abnormal;
    part4_normal;part4_abnormal;
    part5_normal;part5_abnormal];
label=[part1_normal_label;part1_abnormal_label;
    part2_normal_label;part2_abnormal_label;
    part3_normal_label;part3_abnormal_label;
    part4_normal_label;part4_abnormal_label;
    part5_normal_label;part5_abnormal_label];
experiment=1;%实验次数
NC_Fold=zeros(5,experiment);%NC每次实验5折的每一折的精度
ADHD_Fold=zeros(5,experiment);%
%%
for e=1:experiment
    for i=1:5
        test_set_index=[flodindex==i];
        train_set_index=~test_set_index;
        test_x=data(test_set_index,:);
        train_x=data(train_set_index,:);
        train_y=label(train_set_index,:);
        test_y=label(test_set_index,:);
        [train_x,PS]=mapminmax(train_x');%向量归一化到[-1,1]
        test_x=mapminmax('apply',test_x',PS);
        trainData=[train_y train_x']; %合并label和attributes 155*7617
        testData=[test_y test_x'];
        save train.txt -ascii trainData;
        save test.txt -ascii testData;
        disp('*****************');
        [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, TestingAccuracyNC,TestingAccuracyAD] =elm_kernel('train.txt','test.txt', 1,1, 'RBF_kernel',10);
        delete test.txt;
        delete train.txt;
        NC_Fold(i,e)=TestingAccuracyNC;
        ADHD_Fold(i,e)=TestingAccuracyAD;
    end
end
NC_Fold=NC_Fold';
ADHD_Fold=ADHD_Fold';

