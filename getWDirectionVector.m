function [ dI, GInv_Cell_Array] = getWDirectionVector( GInv_Cell_Array, zI, D_Cell_Array, w, q, Active_Columns, Add_Column_Flag, Changed_Column_Index, Changed_Active_Column_Index, Data_Tensor_Dimensions, Tensor_Dimensions, step_size, Precision_Order, GPU_Computing ) 
%getWDirectionVector v1.2
%Author : Ishan Wickramsingha
%Date : 2020/08/24
%Modified Date : 2020/09/23

%getDirectionVector function update GInv for column addition and removal
%using the shur complements inversion formula for column addition and removal.
%Finally calculate the direction vector dI from the updated GInv.
%This function support GPU computation and if the GPU memory is limited the
%function will use less GPU memory for computations.


%% Function Call
%[ dI, GInv] = getDirectionVector( GInv, zI, Gramian_Cell_Array, Active_Columns, Add_Column_Flag, Changed_Column_Index, Changed_Active_Column_Index, Tensor_Dimensions, Precision_Order, GPU_Computing );

%% Inputs 

%Variable        Type           Description

%GInv                       (Matrix)        = Inverse of the Gramian
%zI                         (Numeric Array) = Vector to multiply with GInv
%Gramian_Cell_Array         (Cell Array)    = Contains the 1 dimensional matrices of the separable kronecker Gram matrix of the dictionry: G = D'D 
%Active_Columns             (Numeric Array) = Array containing Active elements of the matrix A to be used in matrix vector calclation
%Add_Column_Flag            (Logical)       = If true a column is added, else a column is removed from the active set          
%Changed_Column_Index       (Numeric)       = Index of the changed column in the separable dictionary D
%Changed_Active_Column_Index(Numeric)       = Index of the changed column in the Active Columns Array
%Tensor_Dimensions          (Logical)       = Dimensions of the tensor as an array
%Precision_Order            (Numeric)       = Order of the precision
%GPU_Computing              (Logical)       = True/False : If True run on GPU


%% Outputs  

%dI   (Numeric Array) = Calculated direction vector : dI = (A'A)\zI
%GInv (Matrix)        = Updated inverse of the Gramian


%% Initialization

N = length(Active_Columns);
order = length(D_Cell_Array);

if isa(GInv_Cell_Array{1},'gpuArray')         
    dI = zeros(N,1,'gpuArray');
    local_gpu_Computing = true;        
else    
    dI = zeros(N,1);
    zI =  gather(zI);
    local_gpu_Computing = false;
end

%% Add Column
    
if Add_Column_Flag
    
    % Add a new cell array if previous ones are full
    if size(GInv_Cell_Array{end},2) < N
        if local_gpu_Computing
            gpu = gpuDevice();
            step_size = min(step_size, max(round(gpu.FreeMemory/ (8 * 10 * (N +step_size - 1))),500));
            max_element_count = step_size*(N +step_size - 1);
                      
            len_GInv = length(GInv_Cell_Array);
            row_dims = zeros(1, len_GInv);
            column_dims = zeros(1, len_GInv);

            for i=1:len_GInv
                [row_dims(i), column_dims(i)] = size(GInv_Cell_Array{i});
            end           
            
            % Divide cell matrices if number of elemtns in cells are larger than element_count
            if sum( row_dims.*column_dims > max_element_count ) > 0
                fprintf('Re-arranging GInv_Cell_Array for element count = %d \n', max_element_count);
                new_GInv_Cell_Array = {};
                for i=1:len_GInv     
                    r_len = row_dims(i);
                    c_len = column_dims(i);

                    if r_len*c_len > max_element_count
                        new_row_length = round(r_len/ceil(r_len*c_len/max_element_count));
                        start_index = 1;
                        while (start_index < r_len)                        
                            end_index = min(start_index + new_row_length - 1, r_len);
                            column_length = c_len - r_len + end_index;
                            new_GInv_Cell_Array{end+1} = GInv_Cell_Array{i}(start_index:end_index,1:column_length);  
                            start_index = end_index + 1;
                        end
                        GInv_Cell_Array{i} = [];
                    else
                        new_GInv_Cell_Array{end+1} = GInv_Cell_Array{i};
                        GInv_Cell_Array{i} = [];
                    end                
                end
                GInv_Cell_Array = new_GInv_Cell_Array;
                clear new_GInv_Cell_Array;                   
            end
            
            fprintf('GInv step size = %d \n', step_size);
            GInv_Cell_Array(end+1) = {zeros(step_size,N + step_size - 1,'gpuArray')};
        else
            fprintf('GInv step size = %d \n', step_size);
            GInv_Cell_Array(end+1) = {zeros(step_size,N + step_size - 1)};
        end
    end                         

    factor_column_indices = getKroneckerFactorColumnIndices( order, Changed_Column_Index, Tensor_Dimensions );
    da = getKroneckerMatrixColumn( D_Cell_Array, factor_column_indices, local_gpu_Computing ); 
    wda = full(w.*da*q(Changed_Column_Index));
    
    Wda = reshape(wda,Data_Tensor_Dimensions); % u= W*da
     
    Ga = fullMultilinearProduct( Wda, D_Cell_Array, true, GPU_Computing );     % V= D'*W*Da   
    ga = q.*vec(Ga);
    
    ga = ga(Active_Columns);

    b = zeros(N,1);
    b(end) = 1;

    if local_gpu_Computing
        b = gpuArray(b);
        ga = full(ga);
    end
    
    len_GInv = length(GInv_Cell_Array);
    row_dims = zeros(1, len_GInv);
    column_dims = zeros(1, len_GInv);
    
    for i=1:len_GInv
        [row_dims(i), column_dims(i)] = size(GInv_Cell_Array{i});
    end      
    
    for i=1:len_GInv
        start_index = min(sum(row_dims(1:i-1))+1, N-1);
        end_index = min(start_index + row_dims(i) - 1, N-1);
        column_end_index = min(column_dims(i), N-1);
        row_end_index = min(min(row_dims(i), N - 1),N-1 - sum(row_dims(1:i-1)));
        
        if row_end_index > 0
            b(start_index:end_index) = b(start_index:end_index) - GInv_Cell_Array{i}(1:row_end_index, 1: column_end_index)*ga(1: column_end_index);  
            b(1: column_dims(i) - row_dims(i)) = b(1: column_dims(i) - row_dims(i)) - (ga(start_index:end_index)'*GInv_Cell_Array{i}(1:row_end_index, 1:column_dims(i) - row_dims(i)))'; 
        end
    end

    alpha = 1 / (ga(N) + (ga(1:N-1)'*b(1:N-1)));
    
    for i=1:len_GInv
        start_index = min(sum(row_dims(1:i-1))+1, N);
        end_index = min(start_index + row_dims(i) - 1, N);
        column_end_index = min(column_dims(i), N);
        row_end_index = min(min(row_dims(i), N),N - sum(row_dims(1:i-1)));
        
        if row_end_index > 0        
            GInv_Cell_Array{i}(1:row_end_index, 1:column_end_index) = GInv_Cell_Array{i}(1:row_end_index, 1:column_end_index) + (alpha*b(start_index:end_index))*b(1:column_end_index)';        

            dI(start_index:end_index) = dI(start_index:end_index) + GInv_Cell_Array{i}(1:row_end_index, 1:column_end_index)*zI(1: column_end_index);  
            dI(1: column_dims(i) - row_dims(i)) = dI(1: column_dims(i) - row_dims(i)) + (zI(start_index:end_index)'*GInv_Cell_Array{i}(1:row_end_index, 1:column_dims(i) - row_dims(i)))';        
        end
    end
    
% Remove Column
else
    len_GInv = length(GInv_Cell_Array);  
    row_dims = zeros(1, len_GInv);
    column_dims = zeros(1, len_GInv);
    
    for i=1:len_GInv
        [row_dims(i), column_dims(i)] = size(GInv_Cell_Array{i});
    end   
    
    alpha = 0;
    ab = zeros(N + 1, 1);
    
    if local_gpu_Computing
        ab = gpuArray(ab);
    end    
    
    for i=1:len_GInv
        start_index = min(sum(row_dims(1:i-1))+1, N + 1);
        end_index = min(start_index + row_dims(i) - 1, N + 1);
        effective_rows = min(row_dims(i), N + 1 - start_index + 1);

        if start_index <= Changed_Active_Column_Index &&  Changed_Active_Column_Index <= end_index
            changed_raw_index = Changed_Active_Column_Index - start_index + 1;
            alpha = GInv_Cell_Array{i}(changed_raw_index, Changed_Active_Column_Index);
            ab(start_index: end_index) = GInv_Cell_Array{i}(1: effective_rows, Changed_Active_Column_Index);
            ab(1:Changed_Active_Column_Index) = GInv_Cell_Array{i}(changed_raw_index, 1: Changed_Active_Column_Index);

            GInv_Cell_Array{i}(changed_raw_index, :) = [];
            GInv_Cell_Array{i}(:, Changed_Active_Column_Index) = [];

        elseif column_dims(i) > Changed_Active_Column_Index
            ab(start_index: end_index) = GInv_Cell_Array{i}(1: effective_rows, Changed_Active_Column_Index);
             GInv_Cell_Array{i}(:, Changed_Active_Column_Index) = [];
        end        
     end   

     ab(Changed_Active_Column_Index) = [];

     for i=1:len_GInv
        [row_dims(i), column_dims(i)] = size(GInv_Cell_Array{i});
     end     

        for i=1:len_GInv
            start_index = min(sum(row_dims(1:i-1))+1, N);
            end_index = min(start_index + row_dims(i) - 1, N);
            column_end_index = min(column_dims(i), N);
            row_end_index = min(min(row_dims(i), N),N - sum(row_dims(1:i-1)));
            
            if row_end_index > 0
                GInv_Cell_Array{i}(1:row_end_index, 1:column_end_index) = GInv_Cell_Array{i}(1:row_end_index, 1:column_end_index) + ((-1 / alpha)*ab(start_index:end_index))*ab(1:column_end_index)';        

                dI(start_index:end_index) = dI(start_index:end_index) + GInv_Cell_Array{i}(1:row_end_index, 1: column_end_index)*zI(1: column_end_index);  
                dI(1: column_dims(i) - row_dims(i)) = dI(1: column_dims(i) - row_dims(i)) + (zI(start_index:end_index)'*GInv_Cell_Array{i}(1:row_end_index, 1:column_dims(i) - row_dims(i)))';             
            end
        end          
end

dI = gpuRound(dI, Precision_Order);

if GPU_Computing && ~isa(GInv_Cell_Array,'gpuArray') 
    dI = gpuArray(dI);        
end  

end