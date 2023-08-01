function [ Vector_Index ] = getVectorIndex( Order, Tensor_Indices, Tensor_Dimensions )
%getVectorIndex v1.0
%Author : Ishan Wickramsingha
%Date : 2019/12/02

%This function calculates the index of an element of a vectorized tensor from tensor indices.
%Assume columns indices are in reverse lexographical order

%% Function Call

% [ Vector_Index ] = getVectorIndex( Order, Tensor_Indices, Tensor_Dimensions );

%% Inputs 

%Variable               Type               Description

%Order              (Numeric)         = Order of the tensor
%Tensor_Indices     (Cell Array)   = Tensor indices  
%Tensor_Dimensions  (Numeric Array)   = Dimensions of the tensor as an array

%% Outputs  

%Vector_Index (Numeric) = Corrosponding vector index of an element of a vectorised tensor 

%% getVectorIndex

if Order ~= length(Tensor_Indices)
    error('Tensor order does not match with the length of the tensor index array');
end

if Order ~= length(Tensor_Dimensions)
    error('Tensor order does not match with the length of the tensor dimensions array');
end

Vector_Index = Tensor_Indices{1};
m = 1;

for i = 2:Order

    m = m.*Tensor_Dimensions(i-1);
    Vector_Index = Vector_Index + (Tensor_Indices{i} - 1).*m;
    
end

end

