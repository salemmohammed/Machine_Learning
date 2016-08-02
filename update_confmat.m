function confmat = update_confmat(confmat,output,GT)
%% Authors: N. Najjar and Salem Alqahtani
% This function updates the confusion matrix. This works only for 2x2
% matrices.
% Inputs: 
%        confmat: a 2x2 matrix. Actual labels in the columns and classifer
%                 output in the rows.
%        output : classifier output. Classees are labeled as -1
%                 and 1
%        GT     : the ground truth labels. Classees are labeled as -1
%                 and 1
% Outputs: 
%        confmat: the updated confusion matrix

%%
idx1 = GT == -1;
confmat(1,1) = sum(output(idx1) == -1);
confmat(2,1) = sum(output(idx1) == +1);

idx2 = GT == +1;
confmat(2,2) = sum(output(idx2) == +1);
confmat(1,2) = sum(output(idx2) == -1);
