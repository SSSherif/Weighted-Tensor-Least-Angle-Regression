function [ X_n ] = getModenMatrix(X, Mode_n, Tensor_Dimensions)
%getModenMatrix v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%getModenMatrix returns the mode-n matrix of the tensor X

%% Function Call

%[ Y_n ] = getModenMatrix(Y, Mode_n, Tensor_Dimensions);

%% Inputs 

%Variable           Type               Description

%X                  (N-D Array)     = Input tensor X
%Mode_n             (Numeric)       = Mode n of the tensor
%Tensor_Dimensions  (Numeric Array) = Dimensions of the tensor as an array

%% Outputs  

%X_n (Matrix) = Resultant Mode-n matrix of the tensor X

%% getModenMatrix

permute_array = 1:length(Tensor_Dimensions);
permute_array(Mode_n) = [];
permute_array = [Mode_n permute_array];

Yn = permute(X,permute_array);

rl = Tensor_Dimensions(Mode_n);
Tensor_Dimensions(Mode_n) = [];
cl = prod(Tensor_Dimensions);

X_n = reshape(Yn,[rl cl]);
            
end

