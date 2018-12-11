function [J_history, Theta1, Theta2] = trainNN(X, y, hidden_layer_size, num_labels, lambda, alpha, iters, Theta1, Theta2)
    input_layer_size = size(X, 2);

    if isnan(Theta1)
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    endif

    m = length(y);
    J_history = zeros(iters, 1);

    for iter = 1:iters

        [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

        Theta1 = Theta1 - alpha * Theta1_grad;
        Theta2 = Theta2 - alpha * Theta2_grad;

        J_history(iter) = J;     % Save the cost J in every iteration 
    end
end