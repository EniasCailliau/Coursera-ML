function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X*theta;
J_cost = 1/(2*m)*sum((hypothesis-y).^2);
theta_quad = theta.^2;
J_regularization = lambda/(2*m)*sum(theta_quad(2:end));
J = J_cost + J_regularization;

grad = X'*1/m*(hypothesis-y);
grad(2:end) = grad(2:end) + lambda/m*theta(2:end);








% =========================================================================

grad = grad(:);

end
