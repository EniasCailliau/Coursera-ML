function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
values_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
best_eval = Inf;

for i=1:length(values_C)
    C_test = values_C(i);
    for j=1:length(values_sigma)
        sigma_test = values_sigma(j);
        model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        predictions = svmPredict(model, Xval);
        current_eval = mean(double(predictions ~= yval));
        if(best_eval > current_eval)
            best_eval = current_eval;
            C = C_test;
            sigma = sigma_test;
        end
    end
end






% =========================================================================

end
