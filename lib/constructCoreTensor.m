function [ X ] = constructCoreTensor( Active_Columns, x, Tensor_Dimensions )
%constructCoreTensor v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%Construct the core tensor X form a non-zero part of the vectorized version(x) of the tensor X.

%% Function Call
%[ X ] = constructCoreTensor( Active_Columns, x, Tensor_Dimensions );

%% Inputs 

%Variable           Type               Description

%Active_Columns   (Numeric Array)    = Array containing indices of the non-zero elements of the tensor X
%x                (Numeric Array)    = Vector to be tensorised (non-zero elements of the tensor X)
%Tensor_Dimensions(Numeric Array)    = Dimensions of the tensor as an array

%% Outputs  

%X (N-D Array) = Result of tensorization of the vector x

%% constructCoreTensor

    column_length = prod(Tensor_Dimensions);    
    vx = zeros(1, column_length );
    vx(Active_Columns) = gather(x);
    if isa(x,'gpuArray')
        vx= gpuArray(vx);
    end  
    
    if length(Tensor_Dimensions) > 1
        X = reshape(vx,Tensor_Dimensions); 
    else
        X = vx';
    end
    
    clear vx;   
end

