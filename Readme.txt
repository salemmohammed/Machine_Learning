
* The dataset is downloaded from UCI machine learning data repository with ground truth labels (available at http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes). As described on the website, the original data contains examples with several features encoded as 0, however, the zero values might be biologically impossible and thus these features might be missing features by assumption. Therefore we remove the examples with zero values on four features including Diastolic blood pressure, Triceps skin fold thickness, 2-Hour serum insulin and Body mass index. Then we have 393 examples in total. Five labelers are created for the dataset by setting their sensitivities =[0.65, 0.67, 0.50, 0.65, 0.30] and specificities =[0.66, 0.67, 0.50, 0.30, 0.65], which are used to decide if a labeler yields a correct label on an either true positive or true negative example.

* Note that the paper "Good Learners for Evil Teachers" assumes the samples are split into disjoint subsets and then each subset of the subjects is only labeled by an individual labeler. However, in real life, nobody knows beforehand which labeler is the evil labeler. There can also be many evil labelers. To facilitate this paper, we randomly split the samples evenly into 5 disjoint subsets and for each subset we use one labelers' labels. You'll need to figure out the evil teacher.

* The dataset has been randomly split into 60% in training set, and the rest for testing

* The data structure contains :

X -- original feature matrix
crowd -- origianl labels from multiple annotators
Y -- labels 
Y_golden -- ground truth
assigned_sampleID -- 5 x 1 cell array, holding the IDs of the examples assigned to a labeler, the order of the cells is the same as that used in the                         sensiticity and specificity array  
X_train, Y_train -- training data
trainSet_sampleID -- ids of samples in training set
X_test, Y_test -- test data
trainSet_sampleID -- ids of samples in test set
sample_labelerID -- 1 x n vector to hold the corresponding labeler ID for each sample, the labeler id corresponds to the order used in the sensiticity                      and specificity array  




#------Training module : please create your own training function that trains a classifier using the method in the paper, please name this function Train_GETeachers as follows:

Function Format

[ Classifier] = Train_GETeachers(X, Y, lambda, Opts)


% Training module : train a classifier using linear SVM according to the paper to learn from multiple annotators

% INPUT:
% X :  n x d feature matrix, each row is a example
% Y :  1 x n labels' matrix, each row is the class labels given by a labeler, Y is +1 or -1
% lambda : the regularization parameter, set as 1 by default
% opts : optional parameters may incldue following
%        w0 : d x 1 vector, for initializing w
%        b0 : the bias
%        epsilon
%        threshold
%        ...
%        and other optional parameters ( specify in the function )

% OUTPUT:
% Classifier : [data structure containing the learnt model parameters including]
%              Classifier.w : d x 1 vector, model parameters
%              Classifier.b : bias
%              Classifier.v : t x 1 vector




#------Testing module : please create a function named below that computes sigmoid of the classifier's output, tests the classifier by drawing the ROC curves, and compute AUC

Note: rocplot matlab function has been provided!!

Function Format

[sigmoid_output] = Test_GETeachers(X, Classifier)

% INPUT :
% X : n x d feature matrix
% Classifier : [data structure containing the learnt model parameters including]
%              Classifier.w : d x 1 vector, model parameters
%              Classifier.b : bias
%              Classifier.v : t x 1 vector
%
% OUTPUT :
% sigmoid_output : translate classifier's output using sigmoid function


Please use the main.m to be your main interface script where you call the above two functions that you created to do the experiments.