%function AUC = rocplot(labels,lineSpec) is used to plot a ROC curve
%and compute the area under the ROC curve.
%Author: Jinbo Bi and Minghu Song (8/27/2003)
%Inputs: labels -- a matrix of 2 columns where the first column is
%			the observed labels (or classes) for each example 
%			and the second column is the predicted values (real
%           numbers) from a probabilitic model for each example
%        lineSpec -- the line specification of the ROC curve, for
%        example, if it is 'r-', the ROC curve will be a red-color
%        solid line, please see MatLab line specification syntax
%        for details.
%Outputs: A ROC curve will be drawn on screen
%         AUC -- area-under-the-curve
%         rates -- [1-spec, sen]
function [AUC,rates] = rocplot(labels,lineSpec,titles,plotSurpress)
n = size(labels,1);
n_pos = sum(labels(:,1)>0);
n_neg = n - n_pos;
if (n_pos==0 || n_neg==0)
	fprintf('contains only one class, and cannot plot ROC curve\n');
    AUC = 0;
else
    [sort_labels,Idx]=sortrows(labels,[2 1]);
    TP=n_pos;
    TN=0;
	
    %compute the sensitivity and specificity numbers	
    j=1;
    Sensitivity(j,1)=1;
    Specificity(j,1)=0;
    for i=1:n
        %i,
        if(sort_labels(i,1)<0)
            TN=TN+1;
        else
            TP=TP-1;
        end
        if( ((i<n)&&(sort_labels(i,2)~=sort_labels(i+1,2)))||(i==n) )
            j=j+1;
            Sensitivity(j,1)=TP/n_pos;
            Specificity(j,1)=TN/n_neg;
        end
    end

    %sort false pos rates and obtain corresponding true pos rates
    rates = [(1-Specificity) Sensitivity];
    rates = sortrows(rates,[1 2]);
    m = size(rates,1);
    j=1; k=1;
    sort_rates(k,1)=rates(j,1);
    while(1)
        I = find(rates(:,1)==sort_rates(k,1));
	   sort_rates(k,2)=max(rates(I,2));
	   j=j+length(I);
	   if (j>m) 
           break;
	   end
	   k=k+1;
	   sort_rates(k,1)=rates(j,1); 
    end
    sort_rates;
    
    if(plotSurpress==false)
        %draw the ROC curve using the sorted rates
        if exist('lineSpec','var') && ~isempty(lineSpec),
            plot(sort_rates(:,1),sort_rates(:,2),lineSpec,'LineWidth',2.5);
            %title('ROC Curve','FontSize',15);
            title(titles,'FontSize',15);
            xlabel('False Positive Rate','FontSize',30);
            ylabel('True Positive Rate','FontSize',30);
            set(gca,'FontSize',30);
        end
    end

    %calculate the area under the ROC curve
    m = size(sort_rates,1);
    up_rates=sort_rates(2:m,:);
    low_rates=sort_rates(1:m-1,:);
    AUC=sum((up_rates(:,1)-low_rates(:,1)).*(up_rates(:,2)+low_rates(:,2))/2);
end