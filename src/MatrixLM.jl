module MatrixLM


using Distributed
using SharedArrays
using Statistics
using LinearAlgebra
using LinearAlgebra.BLAS
import LinearAlgebra.I, LinearAlgebra.mul!, 
    LinearAlgebra.diag, LinearAlgebra.diagm
using DataFrames
using Random
using StatsModels


export Response, Predictors, RawData, get_X, get_Z, get_Y, contr, 
    kron_diag, add_intercept, remove_intercept, shuffle_rows, shuffle_cols, 
    Mlm, mlm, t_stat, coef, predict, fitted, resid, 
    perm_pvals, mlm_perms, design_matrix


# Data object types
include("data_types.jl")
# Contrasts
include("contr.jl")
include("design_matrix.jl")

# Predicted values
include("calc_preds.jl")
# Residuals
include("calc_resid.jl")

# Diagonal of kronecker product
include("kron_diag.jl")
# Variance shrinkage
include("shrink_sigma.jl")

# Miscellenous helper functions
include("misc_helpers.jl")
# Matrix linear model helper functions
include("mlm_helpers.jl")

# Matrix linear models
include("mlm.jl")
# Predictions and residuals
include("predict.jl")

# Permutations
include("perm_pvals.jl")
include("mlm_perms.jl")

end 
