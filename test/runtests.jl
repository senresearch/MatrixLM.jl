###########
# Library #
###########

using Aqua
using MatrixLM, LinearAlgebra, GLM
using DataFrames 
using Random, StatsModels, Statistics, Distributions
using Test


########
# Test #
########


@testset "MatrixLM" begin 
    @testset "Aqua" begin
        Aqua.test_all(MatrixLM)
    end
    include("mlm_test.jl")
    include("misc_helpers_test.jl")
    include("shrink_sigma_test.jl")
    include("mlm_perms_test.jl")
    include("predict_test.jl")
    include("kron_diag_test.jl")
    include("design_matrix_test.jl")
    include("contr_test.jl")
    include("calc_sigma_test.jl")
    include("summary_test.jl")
    include("show_test.jl")
end
