## Weighted Tesnor Least Angle Regression (WT-LARS)

    WTLARS v1.0.0
    Author: Ishan Wickramasingha, Biniyam K. Mezgebo, Sherif S. Sherif
    Date: 2020/08/26
    Modified Date: 2023/08/02

    MATLAB Version: MATLAB R2017b and above

# Abstract:
Sparse weighted multilinear least-squares is a generalization of the sparse multilinear least-squares problem, where prior information about, e.g., parameters and data is incorporated by multiplying both sides of the original problem by a typically diagonal weights matrix. However, the introduction of arbitrary diagonal weights would result in a non-Kronecker least-squares problem that could be very large to store or solve practically. In this paper, we generalize our recent Tensor Least Angle Regression (T-LARS) algorithm to efficiently solve either L0 or L1 constrained multilinear least-squares problems with arbitrary diagonal weights for all critical values of their regularization parameter. To demonstrate the validity of our new Weighted Least Angle Regression (WT-LARS) algorithm, we used it to successfully solve three different image inpainting problems by obtaining sparse representations of binary-weighted images. 

# References

If you use this code in a scientific publication, please cite the following paper:

Ishan M. Wickramasingha, Biniyam K. Mezgebo, and Sherif S. Sherif. "Weighted Tensor Least Angle Regression for Solving Sparse Weighted Multilinear Least Squares Problems" New Approaches for Multidimensional Signal Processing: Proceedings of International Workshop, NAMSP 2023. Springer Nature, 2023.

 Wickramasingha I, Elrewainy A, Sobhy M, Sherif SS. Tensor Least Angle Regression for Sparse Representations of Multidimensional Signals. Neural Comput. 2020;32(9):1-36. doi:10.1162/neco_a_01304 

 ## Example

MATLAB Version: MATLAB R2017b and above

    ./WT-LARS/Example.m

# Function Calls

The WT-LARS function could be called with the following function call.

    [ X ] = WTLARS( Y, D_cell_array, w, Tolerence);

 Also, the WT-LARS supports function calls with more than one available output and three available inputs
 
    [ X, Active_Columns ] = WTLARS( Y, D_cell_array, w, Active_Columns_Limit, ...);
    [ X, Active_Columns, x ] = WTLARS( Y, D_cell_array, w, Active_Columns_Limit, ...);

Following is the complete WT-LARS function call with all available input and output parameters

    [ X, Active_Columns, x, Parameters, Stat, Ax ] = WTLARS( Y, D_Cell_Array, w, Active_Columns_Limit, Tolerence, X, L0_Mode, Mask_Type, GPU_Computing, Plot, Debug_Mode, Path, Iterations, Precision_factor );

# License

See [LICENSE](LICENSE).

# Inputs 
    Variable                  Type         Default         Description

    Y                      (N-D Array)                 = Input data tensor
    D_cell_array           (Cell Array)                = Contains the 1 dimensional dictionarary matrices of the separable dictionary D
    w                      (Numeric Vector)            = Weights as a column vector
    Active_Columns_Limit   (Numeric)                   = Limit of active columns (Depends on the GPU)
    Tolerence              (Numeric)       0.01        = The target residual error as a tolerence to stop the algorithm
    X                      (N-D Array)     0           = Previous partially calculated solution. If X = 0 TLARS runs from the begining.
    L0_mode                (Logical)       False       = True/False : True for L0 or false for L1 Minimization
    Mask_Type              (string/char)   'KP'        = 'KP': Kronecker Product, 'KR': Khatri-Rao Product
    GPU_Computing          (Logical)       True        = True/False : If True run on GPU if available
    Plot                   (Logical)       False       = Plot norm(r) at runtime
    Debug_Mode             (Logical)       False       = True/False : Save TLARS variable into a .mat file given in path in debug mode
    Path                   (string/char)   ''          = Path to save all variables in debug mode
    Iterations             (Numeric)       numel(Y)    = Maximum Number of iteratons to run
    Precision_factor       (Numeric)       10          = Round to 10 times the machine precission 


# Outputs  

    Variable                   Type                Description

    X                      (N-D Array)             = Output coefficients in a Tensor format 
    Active_Columns         (Numeric Array)         = Active columns of the dictionary
    x                      (Numeric Array)         = Output coefficients for active columns in a vector format

    Parameters                                     = Algorithm Parameters class
         Parameters.iterations                     = t : Total number of iterations 
         Parameters.residualNorm                   = norm(r) : Norm of the Residual at the final solution
         Parameters.lambda                         = Lambda  : Lambda value at the final solution
         Parameters.activeColumnsCount             = Number of Active Columns
         Parameters.time                           = Total Time spent

    Stat                                           = WTLARS Statistics Map for each iteration t
         Stat(t).iteration                         = Iteration t
         Stat(t).residualNorm                      = Norm of the residual at iteration t 
         Stat(t).column                            = Changed column at iteration t  
         Stat(t).columnIndices                     = Factor indices of the added column
         Stat(t).addColumn                         = Add a column or remove a column at iteration t   
         Stat(t).activeColumnsCount                = length of the active columns at iteration t 
         Stat(t).delta                             = Delta at iteration t
         Stat(t).lambda                            = Lambda at iteration t
         Stat(t).time                              = Total elapsed time at iteration t
