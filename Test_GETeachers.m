function [output] = Test_GETeachers(X, Classifier)
%% Authors: Nayeff Najjar and Salem Alqahtani 
% Date   : 11/22/2014 Testing

% module : compute sigmoid of the classifier's output, test the classifier
%          by drawing the ROC curves, compute AUC
% 
% INPUT :
%          X : n x d feature matrix 
%          OUTPUT: Classifier : [data structure containing the learnt model
%                       parameters including]
%              Classifier.w : the optimal solution (d x 1 vector)   
%              Classifier.b : the bias 
%              Classifier.alpha : the dual solution (n x 1 vector) 
%              Classifier.threshold : the same threshold in the input
%              Classifier.sv : the support vectors

% 
% OUTPUT : 
%          output : an n x 1 vector (where n is the number of testing
%                   sample points) that contains the decisions of the
%                   classifier
%%

output = ...
    sign(sum((repmat(Classifier.w,size(X,1),1).* X),2)+Classifier.b);
