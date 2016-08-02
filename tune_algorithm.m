function [Comp, NumOfPos_NumOfNeg] = tune_algorithm(data,data_directory,Lambda,...
    NumOfTeachers,Epsilon)

%% Authors: N. Najjar and Salem Alqahtani
% Date: 11/30/2014
% This function performs grid search over all provided combinations of
% tuning parameters for the SVM suggested in the following source:
%      "O. Dekel, and O. Shamir. "Good learners for evil teachers."
%      Proceedings of the 26th annual international conference on machine
%      learning. ACM, 2009."
% "lambda" and "epsilon" drawn from the vectors Lambda and Epsilon.
%
% The function also plots a contour map that shows the accuracy vs the
% variation in lambda and epsilon
% Inputs:
%        data : a data structure with the following fields
%                           X : original feature matrix
%                           crowd : origianl labels from multiple
%                                   annotators
%                           Y : labels
%                           Y_golden : ground truth
%                           assigned_sampleID : 5 x 1 cell array, holding
%                                               the IDs of the examples
%                                               assigned to a labeler
%                           X_train, Y_train : training data
%                           trainSet_sampleID : IDs of samples in training
%                                               set
%                           X_test, Y_test : test data
%                           trainSet_sampleID : IDs of samples in test set
%                           sample_labelerID : 1 x n vector to hold the
%                                              corresponding labeler ID for
%                                              each sample, the labeler id
%                                              corresponds to the order
%                                              used in the sensiticity
%                                              and specificity array
%        data_directory : the directory where the data is
%        Lambda : a vector from which 'lambda' is drawn
%        NumOfTeachers : number of teachers including good and bad ones
%        Epsilon : a vector from which 'epsilon' is drawn
% Outputs:
%        Comp: an n x 3 matrix, where n is the number of testing points.
%              The first column containts the values of lambda while the
%              second column contains values of epsilon. For each pair of
%              lambda and epsilon, the third column of Comp holds the
%              accuracy of classification defined as the area under the ROC
%              curve.
%%

count = 1;
for lambda = Lambda
    for epsilon = Epsilon
        [Classifier] = Train_GETeachers(data, lambda, NumOfTeachers, ...
            epsilon);
        [output] = Test_GETeachers(data.X_test, Classifier);
        
        
        % AUC = sum(output == data.Y_test')/length(output);
        confmat = zeros(2);
        confmat = update_confmat(confmat,output,data.Y_test);
        testBias(count,1) = sum(confmat(1,:)) == length(output) || ...
            sum(confmat(2,:)) == length(output);
        
        %             AUC = (sum(confmat(:,1))/sum(confmat(:))*confmat(1,1) + ...
        %                 sum(confmat(:,2))/sum(confmat(:))*confmat(2,2))/...
        %                 sum(confmat(:));
        X = data.X_test;
        output = sum((repmat(Classifier.w,size(X,1),1).* X),2)+Classifier.b;
%         hold on
%         plot(output);
        AUC...
            = rocplot([data.Y_test' output],'r-','',true);
        
        Comp(count,:) = [lambda, epsilon, AUC];
        NumOfPos_NumOfNeg(count,:)= [lambda, epsilon, ...
            sum(sign(output)>0), ...
            sum((output)<0)];
        count = count +1;
        
    end
end
Comp = Comp(~testBias,:);
%Comp_cell = mat2cell(Comp,ones(1,count-1),ones(1,3));
%Comp_cell = [{'lambda'}, {'epsilon'}, {'accuracy'}; Comp_cell];
% try
%     temp = load('GridSearchOutput.mat');
%     temp = temp.Comp_cell(2:end,:);
% catch
%     temp = [];
% end
% temp = [];
% Comp_cell = [Comp_cell; temp];

%save([data_directory '\GridSearchOutput'],'Comp_cell')

% Comp = cell2mat(Comp_cell(2:end,:));
% Comp = unique(Comp,'rows');
% Comp = sortrows(Comp,[3 1 2]);



figure
hold on
%Lambda_values = unique(Comp(2:end,1));
%count = 1;

intensity_values= unique(Comp(:,3));
Colors = zeros(size(intensity_values,1),3);
L = round(length(intensity_values)/10);
for i =1: length(intensity_values);
    temp = intensity_values(i);
    idx = Comp(:,3) == temp;
%     Colors(i,:) = (1-[.5 1/(1+exp(-2*temp)) 1/(1+exp(-2*temp))]);
    if i <=L
            Colors(i,:)= [.9 1-temp .9];
    elseif i <=2*L
            Colors(i,:)= [.9 .9 1-temp];
    elseif i <=3*L
            Colors(i,:)= [.6 1-temp .6];
    elseif i <=4*L
            Colors(i,:)= [.6 .6 1-temp];
    elseif i <=5*L
            Colors(i,:)= [.3 1-temp .3];        
    elseif i <=6*L
            Colors(i,:)= [.3 .3 1-temp];        
    elseif i <=7*L
            Colors(i,:)= [.1 1-temp .1];        
    elseif i <=8*L
            Colors(i,:)= [.1 .1 1-temp];
    elseif i <=9*L
            Colors(i,:)= [.4 .7 1-temp];        
    else           
            Colors(i,:)= [1-temp 0 0];
    end
    shape = 's';
%     if testBias(i,1)
%         shape= '^';
%     end 
    %Colors(i,:)= [.5 1-temp 1-temp];
    plot(Comp(idx,1),Comp(idx,2),shape,'Color',Colors(i,:),...
        'MarkerFaceColor',Colors(i,:),'MarkerSize',4)
    
    
end
% Colors = 10*Colors;
% Colors = 2.^Colors;
% Colors = Colors./repmat(sum(Colors),size(Colors,1),1);
[max_value, idx_max] = max(Comp(:,3));
idx = Comp(:,3) == max_value;
idx = find(idx);

lambda = Comp(:,1);
eps = Comp(:,2);
%plot(lambda(idx),eps(idx),'+','MarkerSize',7)
intensity_values_cell = num2cell(intensity_values);
intensity_values_cell = cellfun(@(X) num2str(X),...
    intensity_values_cell,'UniformOutput',false);
%legend(intensity_values_cell,'Location','NorthEastOutside');
xlabel('\lambda')
ylabel('\epsilon')

colormap(Colors)
colorbar
caxis([0 max_value]) 
