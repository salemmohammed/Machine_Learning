function [Classifier] = Train_GETeachers(data, lambda, NumOfTeachers,...
                                         varargin)
%% Author: Nayeff Najjar and Salem Alqahtani
% Date: 11/22/2014

% Training module : train a classifier using linear SVM according to the
% paper to learn from multiple annotators

% INPUT: 
%       data : a data structure that contains:
%                           
%
%       lambda : the regularization parameter (default is 1)
%       NumOfTeachers : Total number of teachers including good and bad
%                       ones.
%       Optional parameters may incldue following:
%                        epsilon : a parameter that specifies the influence
%                                  of bad teachers. Default value is 0.1.
%                                  See "O. Dekel, and O. Shamir. "Good
%                                  learners for evil teachers." Proceedings
%                                  of the 26th annual international
%                                  conference on machine learning. ACM,
%                                  2009." for more details about this
%                                  parameter.
%                        w0 : d x 1 vector, for initializing w 
%                        b0 : the bias 
%                        threshold : the threshold is set up so that 
%                        w .* x_i + b > threshold => predicted class is +1
%                        w .* x_i + b < threshold => predicted class is -1

% OUTPUT: Classifier : [data structure containing the learnt model
%                       parameters including]
%              Classifier.w : the optimal solution (d x 1 vector)   
%              Classifier.b : the bias 
%              Classifier.alpha : the dual solution (n x 1 vector) 
%              Classifier.threshold : the same threshold in the input
%              Classifier.sv : the support vectors
%
%  Examples: 
% - To set the value of epsilon, use:
% [...] = Train_GETeachers(data, lambda, NumOfTeachers,epsilon)
%
% - To set both the value of epsilon and the values of w0 and b0, use:
% [...] = Train_GETeachers(data, lambda, NumOfTeachers,epsilon,w0,b0) 
% Note:
% changing the values of w0 and b0 have no effect in the current version
%
% - To set both the value of epsilon and the values of w0 and b0, use:
% [...] = Train_GETeachers(data, lambda, NumOfTeachers,epsilon,w0,b0) 
% Note:
% changing the values of w0 and b0 have no effect in the current version
%
% - To set the threshold, use:
% [...] = Train_GETeachers(data, lambda, NumOfTeachers,epsilon,w0,b0,... 
%               threshold)

%%
X = data.X_train;
Y = data.Y_train';
ID = data.assigned_samplesID;
epsilon = .1;
threshold = 0;
%b = 0; % Temporary set to zeor till the relevant constraints are added.
w0 = X(1,:); 

if length(varargin) ==1
    epsilon = varargin{1};
elseif length(varargin) == 2 
    error('Not enough input parameters')
elseif length(varargin) == 3
    epsilon = varargin{1};
    w0 = varargin{2};
    b0 = varargin{3};
elseif length(varargin) == 4
    epsilon = varargin{1};
    w0 = varargin{2};
    b0 = varargin{3};
    threshold = varargin{4};
end
    

H = 1/lambda * (Y * Y') .* (X*X');
m = length(Y);
f = -1*ones(m,1);
lb = zeros(m,1);
ub = 1/m * ones(m,1);

A_eq = Y';
b_eq = 0;


%% Define Inequality Constraints

A = zeros(NumOfTeachers,m);
b_v = zeros(NumOfTeachers,1);
for i = 1: NumOfTeachers
    temp_ID = ID{i,1};
    [q IDX] = ismember(data.trainSet_sampleID,temp_ID);
    cellSize = sum(q);
    IDX = IDX(IDX~=0);
    A(i,IDX) = 1/cellSize;
    b_v(i,1) = epsilon/(length(Y)*sqrt(cellSize));
end
      



%%
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','iter');
alpha = quadprog(H,f,A,b_v,A_eq,b_eq,lb,ub,w0,opts);
w = sum(repmat(alpha.*Y,1,size(X,2)).*X);

sv_idx = alpha>.5*(max(alpha));
sv = X(sv_idx,:);
b = mean(sum(repmat(w,size(sv,1),1).*sv,2)-Y(sv_idx));


Classifier.w = w;
Classifier.alpha = alpha;
Classifier.b = b;
Classifier.threshold = threshold;
Classifier.sv = sv;