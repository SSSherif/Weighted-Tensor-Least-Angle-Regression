%Run WT-LARS v1.0.0
%Author : Ishan Wickramsingha
%Date : 2022/01/05

%% References

% If you use this code in a scientific publication, please cite the following paper:

% Wickramasingha I, Elrewainy A, Sobhy M, Sherif SS. Tensor Least Angle Regression for Sparse Representations of Multidimensional Signals. Neural Comput. 2020;32(9):1-36. doi:10.1162/neco_a_01304

%%
clear;clc;

data = 'tiger_inpainting'; 
save_data = false;

Y = 0;                              %Data
D_Cell_Array = {};                  %Dictionary matrices(factor matrices) as a cell array
w = 0;                              %Weights as a column vector

Active_Columns_Limit = 10000;       %Limit of active columns (Depends on the required sparsity
Tolerence = 0.05;                   %If Active_Columns_Limit not reached, stop WT-LARS when norm of the residual error reach tolerence)
X = 0;                              %Previous Solution
L0_Mode = false;                    %True for L0 or false for L1 Minimization
Mask_Type = 'KP';                   %'KP': Kronecker Product, 'KR': Khatri-Rao Product 
GPU_Computing = true;               %If True run on GPU if available
Plot = true;                        %Plot norm of the residual at runtime
Debug_Mode = false;                 %Save TLARS variables to a .mat file given in path in the debug mode 
Path = '.\example\';                %Path to save all variables in debug mode

Iterations = 1e6;                   %Maximum Number of iteratons to run
Precision_factor = 10;              %Round to Precision_factor*Machine_Precission
str = '';

algorithm = 'WT-LARS';

%% Prepare Data
Results_Path = strcat(Path,'\results\');
Data_Path = strcat(Path,'\data\');

LP = 'L1';
if L0_Mode
    LP = 'L0';
end

if strcmp(Mask_Type,'KR') 
    product = 'Khatri-Rao';
else
    product = 'Kronecker';
end

%Creating results directories
if save_data || Debug_Mode
    Results_Path = strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'yyyymmdd_HHMM'),'\');
    mkdir(Results_Path);
    diary(strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'.log'));
    diary on
    %profile on
end

% Loading Input Data
load(strcat(Data_Path,data,'.mat'));

fprintf('Running %s %s for %s until norm of the residual reach %d%%  \n\n', algorithm, product, data, Tolerence);
fprintf('Dictionary = %s \n\n',str);

[ X, Active_Columns, x, Parameters, Stat, Ax ] = WTLARS( Y, D_Cell_Array, w, Active_Columns_Limit, Tolerence, X, L0_Mode, Mask_Type, GPU_Computing, Plot, Debug_Mode, Results_Path, Iterations, Precision_factor );

%% Test

if GPU_Computing && gpuDeviceCount == 0
    GPU_Computing = false;
end

D_Cell_Array = cellfun(@(D) normc(D), D_Cell_Array, 'UniformOutput', false);

Y = double(Y);
y = normc(vec(Y));
r = y - Ax;

tensor_norm = norm(vec(Y)); 
Y_reconstructed = gather(reshape(Ax*tensor_norm,size(Y))); 
Ax = gather(Ax);

fprintf('\n %s Completed. \nNorm of the Residual = %g \n',algorithm, norm(r));

%%
if save_data || Debug_Mode
    diary off
    %profile off
    
    f = gcf;
    savefig(f,strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'.fig'));
    saveas(f,strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'.jpg'));
    imwrite(uint8(Y_reconstructed),strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_reconstructed_.jpg'));    
    clear f;
    
    save(strcat(Results_Path,algorithm,'_',LP,'_',data,'_Results','.mat'),'Ax', 'Y_reconstructed','Parameters','Stat','Y','W','D_Cell_Array','Active_Columns', 'x', 'X','-v7.3');
    save(strcat(Results_Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'_GPU.mat'),'-v7.3');    
    %profsave(profile('info'),strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS')));
end



