function [J, grad] = logisticCostFunction(theta, X, y, lambda)

    m = length(y);                % Number of training examples
    J = 0;                        % Set J to the cost
    grad = zeros(size(theta));    % Set grad to the gradient

    len = size(theta, 1);
    lam = zeros(size(theta));
    for i = 2:len
        lam(i) = lambda;
    end

    h_theta = sigmoid(X * theta);
    J = -(1 / m) * (y' * log(h_theta) + (1 - y') * log(1 - h_theta)) + (lambda / (2 * m)) * theta(2:end)' * theta(2:end);
    grad = (1 / m) * X' * (h_theta - y)  + (1 / m) * (lam .* theta);
end