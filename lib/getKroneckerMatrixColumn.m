function [ Column ] = getKroneckerMatrixColumn( Factor_Matrices, Factor_Column_Indices, GPU_Computing )
%getKroneckerMatrixColumn v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%This function return the column of the Kronecker matrix when the factor
%matrices and column indices of each factor matrix is given

%% Function Call

%[ Column ] = getKroneckerMatrixColumn( Factor_Matrices, Factor_Column_Indices, GPU_Computing );

%% Inputs 

%Variable               Type               Description

%Factor_Matrices        (Cell Array)      = Contains the one dimensional matrices of the separable kronecker matrix A 
%Factor_Column_Indices  (Cell Array)      = Contains indices of the one dimensional facor matrices
%GPU_Computing          (Logical)         = True/False : If True run on GPU

%% Outputs  

%Column         (Numeric Array)     = Column of the kronecker Matrix given by Index_Cell_Array

%% getKroneckerMatrixColumn

kronMatrixCount = length(Factor_Column_Indices);
Column = 1; 
column_length = 1;
    
for n = 1:kronMatrixCount

     vec = Factor_Matrices{n}(:,Factor_Column_Indices{n}); 
     
     if GPU_Computing
        vec = gpuArray(full(vec));
     end    
     
     %column = kron(vec,column);
     %kron(A,B) = reshape(bsxfun(@times,A',B),[1 length(A)*length(B)]) %%A,B are column vectors

     vecLength = length(vec);
     column_length = column_length*vecLength; 
     kronProduct = bsxfun(@times,vec',Column);
     Column = reshape(kronProduct,[column_length 1]);
          
end       
end

