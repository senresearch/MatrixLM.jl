module MatrixLM

    using Distributed, SharedArrays
    using Random, Statistics, StatsModels, Distributions
    using LinearAlgebra, LinearAlgebra.BLAS
    import LinearAlgebra.I, LinearAlgebra.mul!, 
           LinearAlgebra.diag, LinearAlgebra.diagm
    using DataFrames
    

    # Data object types
    include("data_types.jl")
    export Response, Predictors, RawData, get_X, get_Z, get_Y

    # Contrasts
    include("contr.jl")
    export contr

    include("design_matrix.jl")
    export  @mlmformula, design_matrix, design_matrix_names

    # Predicted values
    include("calc_preds.jl")

    # Residuals
    include("calc_resid.jl")

    # Diagonal of kronecker product
    include("kron_diag.jl")
    export kron_diag

    # Variance shrinkage
    include("shrink_sigma.jl")

    # Miscellenous helper functions
    include("misc_helpers.jl")
    export add_intercept, remove_intercept,shuffle_rows, shuffle_cols, is_full_rank, check_Z_rank

    # Matrix linear model helper functions
    include("mlm_helpers.jl")
    
    # Matrix linear models
    include("mlm.jl")
    export Mlm, mlm, t_stat

    # Estimate extraction
    include("estimates.jl")
    export get_estimates

    # Predictions and residuals
    include("predict.jl")
    export coef, predict, fitted, resid

    # Permutations
    include("perm_pvals.jl")
    export perm_pvals

    include("mlm_perms.jl")
    export mlm_perms

end 
