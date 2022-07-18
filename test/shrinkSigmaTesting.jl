###########
# Library #
###########
using Test
using MatrixLM
using Statistics

using Random
using LinearAlgebra


###########################
# Generate Simulated Data #
###########################

# Tolerance for tests
tol = 10.0^(-7)
    
# Dimensions of matrices 
n = 100
m = 200
p = 10
q = 20
    
# Generate some matrices.
Random.seed!(4)
X = rand(n,p)
    
m = mean(X, dims=1)
est, varEst = MatrixLM.cov_est(X)
lambda = MatrixLM.shrink_sigma(X, "A")[2]
T = Matrix{Float64}(I, p, p)
lambda2 = sum(varEst) / sum((est-T).^2)

@testset "shrinkSigmaTesting" begin
    @test isapprox(lambda, lambda2, atol=tol)
    # compaire the center function
    @test isapprox(mean(MatrixLM.center(X)), 0, atol=tol)
    # compaire the result of cov_est with cov function
    @test isapprox(est, cov(X), atol=tol)
    # test the size of variance matrices
    @test size(varEst) == (p,p)
    # Test 1. compaire shrink_sigma function
end;