###########
# Library #
###########

using MatrixLM, LinearAlgebra, GLM
using DataFrames 
using Random, StatsModels, Statistics
using Test


########
# Test #
########


@testset "MatrixLM" begin 
    include("mlm_test.jl")
    include("misc_helpers_test.jl")
    include("shrink_sigma_test.jl")
    include("mlm_perms_test.jl")
    include("predict_test.jl")
    include("kron_diag_test.jl")
    include("design_matrix_test.jl")
    include("contr_test.jl")
    include("calc_sigma_test.jl")
end

