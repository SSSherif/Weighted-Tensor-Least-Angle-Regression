function varargout = WTLARS( Y, D_Cell_Array, w, Active_Columns_Limit, varargin )
%WTLARS v1.0.0
%Author : Ishan Wickramasingha, Biniyam Mezgebo, Sherif S. Sherif
%Date : 2020/08/26
%Modified Date : 2022/01/05

%MATLAB Version : MATLAB R2017b and above

%% References

% If you use this code in a scientific publication, please cite the following paper:

% Wickramasingha I, Elrewainy A, Sobhy M, Sherif SS. Tensor Least Angle Regression for Sparse Representations of Multidimensional Signals. Neural Comput. 2020;32(9):1-36. doi:10.1162/neco_a_01304 

%% Function Calls
% The WT-LARS function could be called with the following function call
%[ X ] = WTLARS( Y, D_cell_array, w, Tolerence);

% Also, the WT-LARS supports function calls with more than one available outputs and three available inputs
%[ X, Active_Columns ] = WTLARS( Y, D_cell_array, w, Active_Columns_Limit, ...);
%[ X, Active_Columns, x ] = WTLARS( Y, D_cell_array, w, Active_Columns_Limit, ...);

% Following is the complete WT-LARS function call with all available input and output parameters
%[ X, Active_Columns, x, Parameters, Stat, Ax ] = WTLARS( Y, D_Cell_Array, w, Active_Columns_Limit, Tolerence, X, L0_Mode, Mask_Type, GPU_Computing, Plot, Debug_Mode, Path, Iterations, Precision_factor );

%% Inputs 
%Variable                  Type         Default         Description
%Y                      (N-D Array)                 = Input data tensor
%D_cell_array           (Cell Array)                = Contains the 1 dimensional dictionarary matrices of the separable dictionary D
%w                      (Numeric Vector)            = Weights as a column vector
%Active_Columns_Limit   (Numeric)                   = Limit of active columns (Depends on the GPU)
%Tolerence              (Numeric)       0.01        = The target residual error as a tolerence to stop the algorithm
%X                      (N-D Array)     0           = Previous partially calculated solution. If X = 0 TLARS runs from the begining.
%L0_mode                (Logical)       False       = True/False : True for L0 or false for L1 Minimization
%Mask_Type              (string/char)   'KP'        = 'KP': Kronecker Product, 'KR': Khatri-Rao Product
%GPU_Computing          (Logical)       True        = True/False : If True run on GPU if available
%Plot                   (Logical)       False       = Plot norm(r) at runtime
%Debug_Mode             (Logical)       False       = True/False : Save TLARS variable into a .mat file given in path in debug mode
%Path                   (string/char)   ''          = Path to save all variables in debug mode
%Iterations             (Numeric)       numel(Y)    = Maximum Number of iteratons to run
%Precision_factor       (Numeric)       10          = Round to 10 times the machine precission 

%% Outputs  

%Variable                   Type                Description

%X                      (N-D Array)             = Output coefficients in a Tensor format 
%Active_Columns         (Numeric Array)         = Active columns of the dictionary
%x                      (Numeric Array)         = Output coefficients for active columns in a vector format
%
%Parameters                                     = Algorithm Parameters class
%     Parameters.iterations                     = t : Total number of iterations 
%     Parameters.residualNorm                   = norm(r) : Norm of the Residual at the final solution
%     Parameters.lambda                         = Lambda  : Lambda value at the final solution
%     Parameters.activeColumnsCount             = Number of Active Columns
%     Parameters.time                           = Total Time spent
%
%Stat                                           = WTLARS Statistics Map for each iteration t
%     Stat(t).iteration                         = Iteration t
%     Stat(t).residualNorm                      = Norm of the residual at iteration t 
%     Stat(t).column                            = Changed column at iteration t  
%     Stat(t).columnIndices                     = Factor indices of the added column
%     Stat(t).addColumn                         = Add a column or remove a column at iteration t   
%     Stat(t).activeColumnsCount                = length of the active columns at iteration t 
%     Stat(t).delta                             = Delta at iteration t
%     Stat(t).lambda                            = Lambda at iteration t
%     Stat(t).time                              = Total elapsed time at iteration t


%% WTLARS

tic
addpath(genpath('.\lib'));

%% Validating Input parameters
fprintf('Validating Input Attributes. \n');

algorithm = 'WT-LARS';

%Default Values
Tolerence = 0.01;              %Default Tolerence
X = 0;                         %Initial Solution  = 0
L0_Mode = false;               %Solve the L1 minimization prblem by default 
Mask_Type = 'KP';              %'KP': Kronecker Product, 'KR': Khatri-Rao Product
GPU_Computing = true;          %GPU computing is used if available 
Plot = false;                  %Plot norm of the residual at runtime
Debug_Mode = false;            %Save TLARS interal data
Path = '';                     %Path to save all variables in debug mode
Iterations = numel(Y);         %Maximum Number of iteratons to run
Precision_factor = 10;         %10*eps - Round to 10 times the default machine precision(eps)

%Validate Inputs
validateattributes(D_Cell_Array,{'cell'},{'nonempty'},algorithm,'D_cell_Array',2);

if length(D_Cell_Array)>1
    validateattributes(Y,{'numeric'},{'nonempty','ndims', length(D_Cell_Array)},algorithm,'Y',1);
else
    validateattributes(Y,{'numeric'},{'nonempty'},algorithm,'Y',1);
end

validateattributes(w,{'numeric'},{'nonempty','numel', numel(Y)},algorithm,'w',3);
validateattributes(Active_Columns_Limit,{'numeric'},{'nonempty','positive'},algorithm,'Active_Columns_Limit',3);

tensor_dim_array = size(Y);
tensor_dim_array(tensor_dim_array <= 1) = [];
cellfun(@(Di, dl, idx) validateattributes(Di,{'numeric'},{'nonempty','nrows', dl},algorithm, strcat('Separable Dictionary','',num2str(idx)),2),D_Cell_Array, num2cell(tensor_dim_array),num2cell(1:max(length(tensor_dim_array))));

if nargin >= 4
    X = varargin{1}; 
    validateattributes(Tolerence,{'numeric'},{'nonnegative','<=', 1},algorithm,'Tolerence',4);
end

if nargin >= 5
    X = varargin{2}; 
    validateattributes(X,{'numeric'},{'nonempty'},algorithm,'X',5);
end

if nargin >= 6
    L0_Mode = varargin{3}; 
    validateattributes(L0_Mode,{'logical'},{'nonempty'},algorithm,'L0_Mode',6);
end

if nargin >= 7 
    Mask_Type = varargin{4};
    validateattributes(Mask_Type,{'char','string'},{'nonempty'},algorithm,'Mask_Type',7);
end

if nargin >= 8 
    GPU_Computing = varargin{5};
    validateattributes(GPU_Computing,{'logical'},{'nonempty'},algorithm,'GPU_Computing',8);
end
          
if nargin >= 9 
    Plot = varargin{6};
    validateattributes(Plot,{'logical'},{'nonempty'},algorithm,'Plot',9);
end

if nargin >= 10 
    Debug_Mode = varargin{7};
    validateattributes(Debug_Mode,{'logical'},{'nonempty'},algorithm,'Debug_Mode',10);
end

if nargin >= 11 
    Path = varargin{8};
    validateattributes(Path,{'char','string'},{},algorithm,'Path',11);
end

if nargin >= 12 
    Iterations = varargin{9};
    validateattributes(Iterations,{'numeric'},{'nonempty','positive'},algorithm,'Iterations',12);
end

if nargin >= 13 
    Precision_factor = varargin{10};
    validateattributes(Precision_factor,{'numeric'},{'nonempty','positive'},algorithm,'Precision_factor',13);
end

%% Define Variables
plot_frequency = 1000;  %After every 1000 iterations plot norm_R and image
step_size = 2000;       %Max rows of Gram Matrix part to store in Gram Cell Array
precision = Precision_factor*eps;
precision_order = round(abs(log10(precision)));
is_orthogonal = false;
reconstruct_with_weights = false;

x = 0;
Active_Columns = [];
column_mask_indices = [];

add_column_flag = -1;
changed_dict_column_index = -1;
changed_active_column_index = -1;
total_column_count = -1;
columnOperationStr = 'add';

order = length(D_Cell_Array);
core_tensor_dimensions = zeros(1,order);
gramian_cell_array = cell(1,order);
active_factor_column_indices = cell(1,order);

GInv_Cell_Array = {zeros(step_size,step_size)};

if nargout >= 5
    Stat = containers.Map('KeyType','int32','ValueType','any');
end

%% GPU computing requirments
if GPU_Computing
    if gpuDeviceCount > 0 
        fprintf('GPU Computing Enabled.\n\n');        
        gpu = gpuDevice();        
    else
        fprintf(2,'GPU Device not found. GPU Computing Disabled.\n\n');
        GPU_Computing = false;
    end
end

%% Initialization
fprintf('Initializing %s... \n', algorithm);

% W = S'S
w=double(w);
s = sqrt(w);


% Normalize Y
Y = double(Y);
ys = s.*vec(Y);
Y = reshape(ys,tensor_dim_array);

tensor_norm = norm(vec(Y)); 
Y = Y./tensor_norm;

y = vec(Y); %vec(Y) returns the vectorization of Y
r = y;      %Initial residual r = y;
norm_R = norm(r);

%Normalize each column of mode-n Dictionaries
core_tensor_dimensions = cell2mat(cellfun(@(X) size(X,2),D_Cell_Array,'UniformOutput',false));
total_column_count = prod(core_tensor_dimensions);

% Initialize X_all if requested
if nargout >= 7
    X_all = gpuArray(zeros(Active_Columns_Limit, 1));
end

%Calculate the normalization Matrix Q = diag(q)
fprintf(' \nCalculating the normalization matrix for %d columns. \n', total_column_count);

norm_q = sqrt(vec(fullMultilinearProduct( reshape(w,tensor_dim_array), cellfun(@(X) X.^2,D_Cell_Array,'UniformOutput',false), true, GPU_Computing )));
norm_q(norm_q==0)=1;
q = 1./norm_q;
% q = 1./sqrt(vec(fullMultilinearProduct( reshape(w,tensor_dim_array), cellfun(@(X) X.^2,D_Cell_Array,'UniformOutput',false), true, GPU_Computing )));
clear norm_q;

if reconstruct_with_weights
    q1= q;
    s1 = s;       
else
    q1 = 1./sqrt(vec(fullMultilinearProduct( reshape(ones(length(w),1),tensor_dim_array), cellfun(@(X) X.^2,D_Cell_Array,'UniformOutput',false), true, GPU_Computing )));
    q1(q1==inf) = 1;
    s1=1;
end

%Prepare the column mask vector
if strcmp(Mask_Type,'KR')   
    
    if ~isequal(core_tensor_dimensions,repmat(core_tensor_dimensions(1),1,order))
        exception = MException('D_Cell_Array:MatrixDimensionsMismatch','Column dimensions of the dictionary matrices should be equal for Khatri-Rao Product.');
        throw(exception);
    end
    
    %Obtain valid Khatri-Rao Columns
    kr_columns = 1:getVectorIndex(order,repmat({2},1,order),core_tensor_dimensions)-1:total_column_count;
    
    %column_mask_indices contains ignred columns from the separable dictionary
    column_mask_indices = 1:1:total_column_count;
    column_mask_indices(kr_columns) = []; 
end

%Check for a previous solution X
if nnz(X) > 0
    %If a previsous solution exists start from X 
    fprintf('Start %s calculations using the existing solution \n', algorithm);
    
    wnx = q.*vec( X);
    WNX = reshape(wnx,core_tensor_dimensions);    
        
    %Calculate the residual tensor
    AX = fullMultilinearProduct( WNX, D_Cell_Array, false, GPU_Computing );
    
    axs = s.*vec(AX);
    AXS = reshape(axs,tensor_dim_array);
    R = Y - AXS;
    r = vec(R);
    norm_r_result = norm(r);
    clear AX;    
   
    %Calculate the coeffiecint tensor and vectorize
    rw = s.*vec(R);
    Rw = reshape(rw,tensor_dim_array);
    C = fullMultilinearProduct( Rw, D_Cell_Array, true, GPU_Computing ); % c = B'*r;
    c = gather(vec(C));
    c(column_mask_indices) = 0;   %Apply column mask to the correlation vector
    c = gpuRound(c, precision_order);
    clear R C;
    
    [lambda,changed_dict_column_index] = max(abs(c));
   
    %Find active columns
    x = vec(X); 
    Active_Columns = find(x ~= 0);
    x = x(Active_Columns); 
   
    fprintf('Number of Active_Columns = %d norm(r) = %d lambda = %d \n', length(Active_Columns),norm(r),lambda);
    
    fprintf('Obtaining the inverse of the Weighted Gramian \n');
    GI = getWeightedGramian( D_Cell_Array, w, q, Active_Columns, GPU_Computing );
    GInv_Cell_Array{1} = inv(GI);
    
    clear GI
    
    if Plot
        Ax = s.*kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, x, q, false, GPU_Computing );
        show_wtlars( Ax, Y, tensor_norm, tensor_dim_array );  
    end
    
else
    %If a previsous solution doesn't exists start from the begining
    %Calculate the initial correlation vector c
    fprintf('Calculating the Initial Correlation Vector. \n');
    
    ys = s.*vec(Y);
    Ys = reshape(ys,tensor_dim_array);

    C = fullMultilinearProduct( Ys, D_Cell_Array, true, GPU_Computing ); % c = B'*r;
    c = q.*gather(vec(C));
    c(column_mask_indices) = 0;   %Apply column mask to the correlation vector
    c = gpuRound(c, precision_order);
    clear C;

    [lambda,changed_dict_column_index] = max(abs(c)); %Initial lambda = max(c)
    
    changed_dict_column_index = gather(changed_dict_column_index);
    lambda = gather(lambda);

    % Set initial active column vector
    add_column_flag = 1;
    Active_Columns = changed_dict_column_index;
    changed_active_column_index = 1;  
    
    columnIndices = getKroneckerFactorColumnIndices( order, changed_dict_column_index, core_tensor_dimensions );
    active_factor_column_indices=cellfun(@(x,y) sort(unique([x y])), active_factor_column_indices, columnIndices, 'UniformOutput', false);   
    
    GInv_Cell_Array{1}(1,1) = 1;
end

if GPU_Computing
   D_Cell_Array = cellfun(@(X) gpuArray(full(X)),D_Cell_Array,'UniformOutput',false);
   GInv_Cell_Array{1} = gpuArray(GInv_Cell_Array{1});   
end

%% WT-LARS Iterations

fprintf('Running %s Iterations... \n\n', algorithm);

for t=1:Iterations     
    try
    %% Calculate the inverse of the Gram matrix and update vector v
    
    % Obtain the sign sequence of the correlation of the active columns
    zI = sign(c(Active_Columns));

    %Calculate the inverse of the Gram Matrix for the active set. If A is the active columns matrix then Gramian GI = A'*A
    if length(zI) > 1 && ~is_orthogonal
        try
            [ dI, GInv_Cell_Array ] = getWDirectionVector(GInv_Cell_Array, zI, D_Cell_Array, w, q, Active_Columns, add_column_flag, changed_dict_column_index, changed_active_column_index, tensor_dim_array, core_tensor_dimensions, step_size, precision_order, GPU_Computing );
            
            if any(isnan(dI))  
                fprintf('The Inverse of the Gramian is singular. Re-Calculating the Inverse of the Gramian \n');
                GI = getWeightedGramian( D_Cell_Array, w, q, Active_Columns, GPU_Computing );
                
                if det(GI) < 1e-50
                    fprintf('The Re-Calculated Gramian is singular. Calculating the Pseudo-Inverse of the Gramian \n');
                    GInv = pinv(GI); %Gramian is singular. Calculating the Pseudo-Inverse of the Gramian
                else
                    GInv = inv(GI);
                end
                if GPU_Computing
                    GInv_Cell_Array = {gpuArray(GInv)};
                else                    
                    GInv_Cell_Array = {GInv};
                end
                dI = GInv_Cell_Array{1}*zI;
                clear GInv GI;
            end             
            
        catch e        
            % In casee of an exception, gather GInv from GPU
            %rethrow(e);
            if GPU_Computing
                fprintf(2,'Exception Occured. Disabling GPU Computing.\nException = %s \n', getReport(e));
                GPU_Computing = false;
                D_Cell_Array = cellfun(@(X) gather(X),D_Cell_Array,'UniformOutput',false);
                GInv_Cell_Array = cellfun(@(X) gather(X),GInv_Cell_Array,'UniformOutput',false);
                [x,c,r,q,q1,v,dI,zI,lambda,Ax,norm_r_result,norm_R]=gather(x,c,r,q,q1,v,dI,zI,lambda,Ax,norm_r_result,norm_R);
                [ dI, GInv_Cell_Array ] = getWDirectionVector(GInv_Cell_Array, zI, D_Cell_Array, w, q, Active_Columns, add_column_flag, changed_dict_column_index, changed_active_column_index, tensor_dim_array, core_tensor_dimensions, step_size, precision_order, GPU_Computing );
            end
        end  
    elseif is_orthogonal
        dI = zI;
    else
        GInv_Cell_Array{1}(1,1) = 1;
        dI = zI;
    end

    %Create vector v by selecting equivalent active columns from the Gramian G and multiplying with dI 
    
     u = kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, active_factor_column_indices, dI, q, false, GPU_Computing );% u= A*dI
     uw = w.*u;
     Uw = reshape(uw,tensor_dim_array); % u= W*A*dI
     
     V = fullMultilinearProduct( Uw, D_Cell_Array, true, GPU_Computing );     % V= D'*W*A*dI
     v = q.*vec(V); 
     v(column_mask_indices) = 0; 
     v = gpuRound(v, precision_order);
     v(Active_Columns) = sign(v(Active_Columns));
     
%% calculate delta_plus and delta_minus for every column 
    
    changed_dict_column_index = -1;
    delta  = -1;
    add_column_flag = 0;

    % Calculate delta_plus 
    delta_plus_1                            = bsxfun(@minus, lambda, c)./bsxfun(@minus, 1, v);    
    delta_plus_1(Active_Columns)            = inf;
    delta_plus_1(column_mask_indices)       = inf;
    delta_plus_1(delta_plus_1 <= precision) = inf;
    [min_delta_plus_1, min_idx1]            = min(delta_plus_1);

    delta_plus_2                            = bsxfun(@plus, lambda, c)./bsxfun(@plus, 1, v);
    delta_plus_2(Active_Columns)            = inf;
    delta_plus_2(column_mask_indices)       = inf;
    delta_plus_2(delta_plus_2 <= precision) = inf;
    [min_delta_plus_2, min_idx2]            = min(delta_plus_2);

    if min_delta_plus_1 < min_delta_plus_2
        changed_dict_column_index = gather(min_idx1);
        delta = full(min_delta_plus_1);
        add_column_flag = 1;
    else
        changed_dict_column_index = gather(min_idx2);
        delta = full(min_delta_plus_2);
        add_column_flag = 1;
    end      

    % Calculate delta_minus for L1 minimization
    if ~L0_Mode
        delta_minus                             = -x./dI;
        delta_minus(delta_minus <= precision)   = inf;
        [min_delta_minus, col_idx3]             = min(delta_minus);
        min_idx3                                = Active_Columns(col_idx3);

        if length(Active_Columns) > 1 && min_delta_minus < delta
            changed_dict_column_index = gather(min_idx3);
            delta = full(min_delta_minus);
            add_column_flag = 0;
        end           
    end        
    
    
    delta = gpuRound(delta,precision_order);

    %% Compute the solution x and parameters for next iteration

    % Check for invalid conditions
    if lambda < delta || lambda < 0 || delta < 0    
        fprintf('%s Stopped at: t = %d  norm(r) = %d lambda = %d  delta = %d Time = %.3f\n',algorithm,t,nr,lambda,delta,toc);
        break;
    end

    %Update the solution x and lambda
    x = x + delta*dI;    
    lambda = gpuRound(lambda - delta, precision_order);  %lambda = max(c(active_columns));   
    c = c - delta*v; %c = B'*r;
    
    % Clear precession errors for c(Active_Columns) by setting it to z*Lambda
    c(Active_Columns) = lambda*sign(c(Active_Columns));

    %Update the norm of the residual
    ad = gather(kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, active_factor_column_indices, dI, q, false, GPU_Computing));%v = A*dI; 
    r = r - delta*s.*ad; %r = r - delta*S*A*dI;
    nr = norm(r);      
    norm_R = [norm_R; nr];  
   
    %Update the Stat class
    if nargout >= 5
        Stat(t) = StatClass( t,gather(nr),changed_dict_column_index,columnIndices,add_column_flag,length(Active_Columns),gather(delta),gather(lambda),toc );
    end   
    
    if nargout >= 7
        X_all(t,:) = x;
    end
    
    %Plot the current result
    if Plot && ( Debug_Mode || rem(t,plot_frequency) == 1 )
        Ax = s1.*kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, active_factor_column_indices, x, q1, false, GPU_Computing );
        show_wtlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array );  
    end   
    
    %Stopping criteria : If the stopping criteria is reached, stop the program and return results
    if nr < Tolerence || length(Active_Columns) >= Active_Columns_Limit
        fprintf('\n%s stopping criteria reached \n%s Finished at: t = %d norm(r) = %d lambda = %d  delta = %d tolerence = %d Time = %.3f\n',algorithm,algorithm,t,nr,lambda,delta,Tolerence,toc);
        break;
    end
    
    %% Add or remove column from the active set
    
    if  add_column_flag            
        Active_Columns = [Active_Columns; changed_dict_column_index];
        x = [x; 0];
        changed_active_column_index = length(x);
        columnOperationStr = 'add';
        if nargout >= 7
            X_all(end,changed_active_column_index) = 0;
        end         
    else
        changed_active_column_index = find(Active_Columns == changed_dict_column_index);
        x(changed_active_column_index)  = [];
        Active_Columns(changed_active_column_index) = [];
        columnOperationStr = 'remove';
        if nargout >= 7
            X_all(:,changed_active_column_index) = [];
        end           
    end    
    
    % Calculate column indices of each changed column
    columnIndices = getKroneckerFactorColumnIndices( order, changed_dict_column_index, core_tensor_dimensions );
    active_factor_column_indices=cellfun(@(x,y) sort(unique([x y])), active_factor_column_indices, columnIndices, 'UniformOutput', false);   
    
 %% Display information   
    
    gpu_usage = "";
    
    if GPU_Computing
        str = 'gpu';        
        if ~isa(GInv_Cell_Array{1},'gpuArray')            
            str = "Limited " + str;
        end
        gpu_usage = sprintf("gpu= %.2f%%",((gpu.TotalMemory-gpu.FreeMemory)*100)/gpu.TotalMemory);
    else        
        str = 'cpu';        
    end
     
    if Debug_Mode 
        fprintf('%s %s t= %d norm(r)= %d active columns= %d indices= %s %s column= %d %s Time= %.3f\n', algorithm, str, t,nr, length(Active_Columns), join(string(columnIndices)),columnOperationStr,changed_dict_column_index,gpu_usage,toc);
    else
        fprintf('%s %s t= %d norm(r)= %d active columns= %d %s column= %d %s Time= %.3f\n',algorithm, str, t,  nr, length(Active_Columns), columnOperationStr, changed_dict_column_index, gpu_usage, toc);        
    end

    %Handle exception
    catch e
        fprintf(2,'Exception Occured.\nException = %s \n', getReport(e));
        if t >1
            
            X = constructCoreTensor(Active_Columns, x, core_tensor_dimensions);
            [x,X,Active_Columns,~,lambda,activeColumnsCount]=gather(x,X,Active_Columns,nr,lambda,length(Active_Columns));
            
            Ax = s1.*kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, active_factor_column_indices, x, q1, false, GPU_Computing );
            
            if Debug_Mode && isdir(Path)
                
                [c,r,v,dI,zI,GInv_Cell_Array,delta,Ax,norm_r_result,norm_R]=gather(c,r,v,dI,zI,GInv_Cell_Array,delta,Ax,norm_r_result,norm_R);
                [delta_plus_1,delta_plus_2,delta_minus]=gather(delta_plus_1,delta_plus_2,delta_minus);
                [min_delta_plus_1,min_delta_plus_2,min_delta_minus]=gather(min_delta_plus_1,min_delta_plus_2,min_delta_minus);
                [min_idx1,min_idx2,min_idx3,col_idx3]=gather(min_idx1,min_idx2,min_idx3,col_idx3);            
                
                save(strcat(Path,algorithm,'_error_at_',num2str(t),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'_Results','.mat'),'Ax','Stat','Y','D_Cell_Array','Active_Columns', 'x', 'X','Parameters','-v7.3');
                save(strcat(Path,algorithm,'_error_at_',num2str(t),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'.mat'),'-v7.3');
            end
            if Plot
                show_wtlars( Ax, Y, tensor_norm,  norm_R, tensor_dim_array ); 
            end
        end                 
        rethrow(e)
    end

end

%% Preparing WT-LARS Output

% construct the core tensor
X = constructCoreTensor(Active_Columns, x, core_tensor_dimensions);
[x,X,Active_Columns,nr,lambda,activeColumnsCount]=gather(x,X,Active_Columns,nr,lambda,length(Active_Columns));

if Plot || Debug_Mode || nargout >= 6
    Ax = s1.*kroneckerMatrixWeightedPartialVectorProduct( D_Cell_Array, Active_Columns, active_factor_column_indices, x, q1, false, GPU_Computing );  
end

% Set output variables based on the number of requested outputs
if nargout >=1
    varargout{1} = X;
end
if nargout >=2
    varargout{2} = Active_Columns;
end
if nargout >=3
    varargout{3} = x;
end
if nargout >=4    
    varargout{4} = Parameters(t,nr,lambda,activeColumnsCount,toc);
end
if nargout >=5
    varargout{5} = Stat;
end
if nargout >=6    
    varargout{6} = Ax;
end
if nargout >=7   
    varargout{7} = X_all; 
end

if Plot
    show_wtlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array );
end

if Debug_Mode && isdir(Path)
    
    [c,r,v,dI,zI,GInv_Cell_Array,delta,Ax,norm_r_result,norm_R]=gather(c,r,v,dI,zI,GInv_Cell_Array,delta,Ax,norm_r_result,norm_R);
    [delta_plus_1,delta_plus_2,delta_minus]=gather(delta_plus_1,delta_plus_2,delta_minus);
    [min_delta_plus_1,min_delta_plus_2,min_delta_minus]=gather(min_delta_plus_1,min_delta_plus_2,min_delta_minus);
    [min_idx1,min_idx2,min_idx3,col_idx3]=gather(min_idx1,min_idx2,min_idx3,col_idx3);
    
    save(strcat(Path, algorithm,'_Finished at_',num2str(t),'.mat'),'-v7.3');
end
    
end

