function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);                               % Setup some useful variables
J = 0;                                        % You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% PART 1: FEED FORWARD PROPAGATION
%eye_matrix = eye(num_labels);
%y_matrix = eye_matrix(y,:); % m x num_labels

# feed-forward pass
a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = [ones(m, 1) sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

y_all = bsxfun(@eq, y, 1:num_labels);
J = (-1/m) * sum(sum(y_all .* log(a_3) + (1 - y_all) .* log(1 - a_3)));

% Add regularization term
J += (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));

% PART 2: BACK PROPAGATION

d_3 = a_3 - y_all;
d_2 = d_3 * Theta2(:, 2:end) .* sigmoidGradient(z_2);
Delta1 = d_2' * a_1;
Delta2 = d_3' * a_2;
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

% PART 3: WEIGHT REGULARIZATION
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1_grad += Theta1 * (lambda / m);
Theta2_grad += Theta2 * (lambda / m);

% ============================================================

grad = [Theta1_grad(:) ; Theta2_grad(:)];     % Unroll gradients

end