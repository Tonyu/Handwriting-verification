clear;
close all;
clc;
input_layer_size=400; %20*20 Input images of digit
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

%selecting 100 datas randomly
select = randperm(size(X, 1));
select = select(1:100);

displayData(X(select, :));

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n'], J);