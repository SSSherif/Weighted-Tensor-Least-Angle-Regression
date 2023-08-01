function [ X ] = fullMultilinearProduct( X, Factor_Matrices, Use_Transpose, GPU_Computing )
%fullMultilinearProduct v1.3
%Author : Ishan Wickramsingha
%Date : 2020/10/31
%Modified : 2020/08/25

%Fullmultilinear product of the tensor X with factor matrices in the Factor_Matrices using both GPU computing and CPU computing.
%If Use_Transpose is true use the transpose of each factor matrix for the kronekcer product.

%% Function Call

%[ X ] = fullMultilinearProduct( X, Factor_Matrices, Use_Transpose, GPU_Computing)

%% Inputs 

%Variable        Type               Description

%X               (N-D Array)    = Input core tensor X
%Factor_Matrices (Cell Array)   = Contains the 1 dimensional matrices of the separable kronecker matrix A 
%Use_Transpose   (Logical)      = Obtain a kronecker product of transposed factor matrices
%GPU_Computing   (Logical)      = True/False : If True run on GPU

%% Outputs  

%X (N-D Array) = Result of the full multilinear product of the input tensor X and each factor matrix.

%% fullMultilinearProduct

if isempty(Use_Transpose)
   Use_Transpose = 0;
end

if isempty(GPU_Computing)
   GPU_Computing = false;
end

if isa(X,'sptensor') ||  isa(X,'tensor') 
    X = double(X);
end

sz = [];
szx = size(X);
order  = length(szx);
factor_count = length(Factor_Matrices);

if Use_Transpose
    sz = cellfun(@(x) [sz size(x,1)], Factor_Matrices);   
else
    sz = cellfun(@(x) [sz size(x,2)], Factor_Matrices);
end

if order ~= factor_count
    if length(sz(sz > 1)) == length(szx(szx > 1)) 
        order = factor_count;
    else
        error('Tensor order does not match with the number of factor matrices');
    end
end 

if GPU_Computing && ~isa(X,'gpuArray') 
    X = gpuArray(X);
end

for n=1:factor_count
    
    Bn = Factor_Matrices{n}; 
    size_Bn = size(Bn);
    
    if GPU_Computing && ~isa(Bn,'gpuArray') 
        Bn = gpuArray(Bn);
    end    

    %Calculate Tensor mode-n matrix     
    if order > 1
        column_order = [n 1:n-1 n+1:factor_count];
        X = permute(X,column_order);
    end
    
    Xn = reshape(X,[sz(n), prod([sz(1:n-1) sz(n+1:end)])]);
        
    %Tensor mode-n Product 
    if Use_Transpose
        Xn = Bn'*Xn;
        sz(n) = size_Bn(2);
    else        
        Xn = Bn*Xn;
        sz(n) = size_Bn(1);
    end
    
    %Obtain N-D Tensor
        
    if order > 1
        X = reshape(Xn,[sz(n) sz(1:n-1) sz(n+1:end)]);
        column_order = [2:n 1 n+1:factor_count];
        X = permute(X,column_order);
    else
        X = reshape(Xn,[sz(n) 1]);
    end
    
end
 
end

