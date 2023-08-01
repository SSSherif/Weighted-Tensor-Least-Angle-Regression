function varargout = weightedCoherence( Factor_Matrices, w, GPU_Computing )
%weightedCoherence v1.0
%Author : Ishan Wickramsingha
%Date : 2020/09/11

%weightedCoherence function calculates the maximum coherence of the weighted knronecker matrix 

%% Function Call

%[ maximum_coherence ] = weightedCoherence( Factor_Matrices, w, GPU_Computing );

%% Inputs 

%Variable        Type           Description

%Factor_Matrices (Cell Array)   = Contains factor matrices of the separable kronecker matrix
%w  (Numeric Array)= Array containing column weights
%GPU_Computing   (Logical)      = True/False : If True run on GPU


%% Outputs  

%maximum_coherence (Numeric) = Result vector of y = Ax; or y = A'x
%max_coherence_column1 (Numeric) = First column with the maximum coherence
%max_coherence_column2 (Numeric) = Second column with the maximum coherence

%% weightedCoherence

kronMatrixCount = length(Factor_Matrices);
factor_columns_dimensions = cell2mat(cellfun(@(X) size(X,2),Factor_Matrices,'UniformOutput',false));
tensor_dim_array = cell2mat(cellfun(@(X) size(X,1),Factor_Matrices,'UniformOutput',false));

maximum_coherence = 0;
max_coherence_column1 = -1;
max_coherence_column2 = -1;

%Obtain 1/normalization values as a vector
q = 1./sqrt(vec(fullMultilinearProduct( reshape(w,tensor_dim_array), cellfun(@(X) X.^2,Factor_Matrices,'UniformOutput',false), true, GPU_Computing )));

for k = 1:prod(factor_columns_dimensions)
    
    factor_column_indices = getKroneckerFactorColumnIndices( kronMatrixCount, k, factor_columns_dimensions );
    column = getKroneckerMatrixColumn( Factor_Matrices, factor_column_indices, GPU_Computing );  
    wka = full(w.*column*q(k));
    
    Wka = reshape(wka,tensor_dim_array);
     
    Gk = fullMultilinearProduct( Wka, Factor_Matrices, true, GPU_Computing );
    gk = q.*vec(Gk);
    gk(k) = 0;    
    
    [coherence, coherence_row ] = max(abs(gk));
    
    if coherence > maximum_coherence
        max_coherence_column1 = coherence_row;
        max_coherence_column2 = k;
        maximum_coherence = coherence;
    end
    fprintf('Column = %d coherence = %d maximum_coherence = %d \n', k, coherence,  maximum_coherence);
end

if nargout >=1
    varargout{1} = maximum_coherence;
end
if nargout >=2
    varargout{2} = max_coherence_column1;
end
if nargout >=3
    varargout{3} = max_coherence_column2;
end

end

