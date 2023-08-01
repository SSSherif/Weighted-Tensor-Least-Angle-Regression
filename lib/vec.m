function [ x ] = vec( X )
%vec v1.0
%Author : Ishan Wickramsingha
%Date : 2019/10/18

%This function obtain a vectorization of a tensor X

%% Function Call

% [ x ] = vec( X );

%% Inputs 

%Variable        Type               Description

%X          (N-D Array)             = Input tensor

%% Outputs  

%x (Array) = Vectorization of the tensor X

%% getTensorIndices

len = prod(size(X));

if isa(X,'sptensor') 
    x = reshape(double(sptenmat(X,1)),[len 1]);
elseif isa(X,'tensor')
    x = reshape(double(X),[len 1]);
else
    x = reshape(X,[len 1]);
end

end