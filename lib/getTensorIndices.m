function [ Tensor_Indices ] = getTensorIndices( Order, Index, Tensor_Dimensions )
%getTensorIndices v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%This function calculates the corrosponding tensor indices of a vector index of an element of a vectorised tensor.
%Assume columns indices are in reverse lexographical order

%% Function Call

% [ Tensor_Indices ] = getTensorIndices( Order, Index, Tensor_Dimensions );

%% Inputs 

%Variable               Type               Description

%Order              (Numeric)         = Order of the tensor
%Index              (Numeric)         = Vector index of an element of a vectorised tensor  
%Tensor_Dimensions  (Numeric Array)   = Dimensions of the tensor as an array

%% Outputs  

%Tensor_Indices (Cell Array) = Corrosponding tensor indices of the vector index of an element of a vectorised tensor 

%% getTensorIndices

mode = 1;


index_array = [];

if Order ~= length(Tensor_Dimensions)
    error('Tensor order does not match with the length of the tensor dimensions array');
end

if Index < 1 || Index > prod(Tensor_Dimensions(1:end))
    error('Invalid Vector index');
end

if mode > 1
    
    if mode > Order
        warning('Invalid mode');
    else        
        Tensor_Dimensions = [Tensor_Dimensions(mode), Tensor_Dimensions(1:end ~= mode)];
    end
end

for i = Order:-1:1
   
    index_array(i) = Index/prod(Tensor_Dimensions(1:i-1));
    
    for j = Order-1:-1:i
        index_array(i) = index_array(i) - (index_array(j+1)-1).*prod(Tensor_Dimensions(i:j));
    end
    
    index_array(i) = ceil(index_array(i));
    
end

if mode > 1
    
    if mode > Order
        warning('Invalid mode');
    else        
        index_array = [index_array(2:mode),index_array(1),index_array(mode+1:end)];
    end
end

Tensor_Indices =  num2cell(index_array);

end

