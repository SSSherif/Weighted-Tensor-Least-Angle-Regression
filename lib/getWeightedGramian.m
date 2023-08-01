function [ A ] = getWeightedGramian (Factor_Matrices, w, q, Active_Columns, GPU_Computing)
%getWeightedGramian v1.0
%Author : Ishan Wickramsingha
%Date : 2020/09/01

%Obtain a Gram matrix by selecting a subset of columns from a
%kronecker matrix given by kronecker product of matrices in the Kron_Cell_Array

%% Function Call
%[ X ] = getGramian(Factor_Matrices, Active_Columns, GPU_Computing);

%% Inputs 

%Variable        Type               Description

%Factor_Matrices (Cell Array)   = Contains the 1 dimensional matrices of the separable kronecker matrix 
%Active_Columns  (Numeric Array)= Array containing indices of selected columns
%GPU_Computing   (Logical)      = True/False : If True run on GPU

%% Outputs  

%A (Matrix) = Resultant kronecker sub matrix

%% getKroneckerSubMatrix

A = zeros(length(Active_Columns));

kronMatrixCount = length(Factor_Matrices);
dim_array = zeros(1,kronMatrixCount);
Data_Tensor_Dimensions = zeros(1,kronMatrixCount);
for n = 1:kronMatrixCount
    dim_array(n) = size(Factor_Matrices{n},2);
    Data_Tensor_Dimensions(n) = size(Factor_Matrices{n},1);
end

qa = q(Active_Columns);

for i = 1:length(Active_Columns)
    
    factor_column_indices = getKroneckerFactorColumnIndices( kronMatrixCount, Active_Columns(i), dim_array );
    da = getKroneckerMatrixColumn( Factor_Matrices, factor_column_indices, GPU_Computing );  
    wda = full(w.*da*qa(i));
    
    Wda = reshape(wda,Data_Tensor_Dimensions);
     
    Gk = fullMultilinearProduct( Wda, Factor_Matrices, true, GPU_Computing );
    g_k = q.*vec(Gk);
    
    gramian_row = g_k(Active_Columns);    
    A(i,:) = gather(gramian_row);
    
end
end

