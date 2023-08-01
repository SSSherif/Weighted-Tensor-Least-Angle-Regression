function [ Factor_Column_Indices ] = getKroneckerFactorColumnIndices( Order, Column_Index, Tensor_Dimensions )
%getKroneckerFactorColumnIndices v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%This function calculates the factor matrix indices from a column index of a kronecker product matrix.
%Assume columns indices are in reverse lexographical order

%% Function Call

%[ Factor_Column_Indices ] = getKroneckerFactorColumnIndices( Order, Column_Index, Tensor_Dimensions );

%% Inputs 

%Variable           Type               Description

%Order              (Numeric)         = Order of the core tensor/ Number of factor matrices
%Column_Index       (Numeric)         = Column index of the kronecker matrix  
%Tensor_Dimensions  (Numeric Array)   = Dimensions of the core tensor as an array

%% Outputs  

%Factor_Column_Indices (Cell Array)  = Clumn indices of each factor matrix

%% getKroneckerFactorColumnIndices

index_array = [];
mode = 1;

if Order ~= length(Tensor_Dimensions)
    error('Tensor order does not match with the length of the tensor dimensions array');
end

if Column_Index < 1 || Column_Index > prod(Tensor_Dimensions(1:end))
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
   
    index_array(i) = Column_Index/prod(Tensor_Dimensions(1:i-1));
    
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

Factor_Column_Indices =  num2cell(index_array);

end

