clc
clear all
close all
%% Global Variables
data_directory = cd; % The directory where the data is 
load([data_directory '\Pima_synthetic.mat']);

NumOfClasses = 2; % Number of Classes
%classSize = 100;
plot_flag = false; % This flag specifies whether the a plot of the data and
                   % the SVM hyperplane is required

lambda = .707; % Regularization parameter;
epsilon = 9; % This controls the influence of bad and good teachers 


NumOfTeachers = 5;
%% Applying the Sigmoid function
data.X_train = (1+exp(-data.X_train)).^(-1);
data.X_test = (1+exp(-data.X_test)).^(-1);


%% Tuning
% Tuning is performed using a grid search. A plot has been generated that
% shows the values that were usd for lambda and epsilon. Maximum accuracy
% was 69% for lambda = 0.707 and epsilon = 9.

close all
Lambda = 1:.5:100;%:1:15];
Epsilon = .01:.01:1;%:-5];%:15];
[Comp,NumOfPos_NumOfNeg] = ...
    tune_algorithm(data,data_directory,Lambda,NumOfTeachers,Epsilon);
% idx = ismember(NumOfPos_NumOfNeg(:,3:4),[157 0; 0 157],'rows');
% NumOfPos_NumOfNeg = NumOfPos_NumOfNeg(~idx,:);
% Comp = Comp(~idx,:);
%xlim([min(Lambda) max(Lambda)])

%ylim([min(Epsilon) max(Epsilon)])

%% Training the Classifier 
[x,idx_max] = max(Comp(:,3));
lambda = Comp(idx_max,1);
epsilon = Comp(idx_max,2);
% lambda = 4;
% epsilon = .97;
[Classifier] = Train_GETeachers(data, ...
                      lambda, NumOfTeachers, ...
                      epsilon);


%% Classifier Testing
[output] = Test_GETeachers(data.X_test, Classifier);

output = -output; %Flip Decision
confmat = zeros(2);
confmat = update_confmat(confmat,output,data.Y_test);

% confmat = zeros(2);
% confmat = update_confmat(confmat,data.Y_test,data.Y_test);






%% ROC
figure
lineSpec = 'r-';
title = 'ROC';
plotSurpress = false;
% [output] = Test_GETeachers(data.X_test, Classifier);
X = data.X_test;
output = sum((repmat(Classifier.w,size(X,1),1).* X),2)+Classifier.b;
AUC...
        = rocplot([data.Y_test' output],lineSpec,title,plotSurpress);
h = refline(1,0);
