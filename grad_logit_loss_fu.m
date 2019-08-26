function grad= grad_logit_loss_fu(A, b, x, L)
%LOGISTIC LOSS
% [objv, grad] = logit_loss(Y, X, w, l2)
% Y     : label
% X     : data
% w     : weight
% l2    : l2 regularizer
% objv  : the objective value
% grad  : the gradient
tau = b .* (A * x);
% tau = max(-100, min(100, tau));
% objv = sum(log(1 + exp(-tau)));
% grad = X' * (- Y ./ (1  + exp(tau))) + l2 * w;
% objv = sum(log(1 + exp(-tau))) +  lamb * norm(x,2)^2;
grad = A' * (- b ./ (1  + exp(tau))) +  L * x;