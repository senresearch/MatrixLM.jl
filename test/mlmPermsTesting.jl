###########
# Library #
###########
using Test
using MatrixLM

using DataFrames
using Random
using LinearAlgebra
using GLM


###########################
# Generate Simulated Data #
###########################

# Tolerance for tests
tol = 50.0^(-2)
    
# Dimensions of matrices 
n = 100
m = 200
p = 10
q = 20
    
# Generate some matrices.
Random.seed!(4)
X = rand(n,p)
Z = rand(m,q)
B = rand(1:20,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E
    
# Data frame to be passed into lm
GLMData = DataFrame(hcat(vec(Y), kron(Z,X)), :auto)
# lm estimate
GLMEst = lm(Matrix(GLMData[:,2:end]), Vector(GLMData[:,1]))

# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)
MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)

nPerms = 5
tStats, pVals = mlm_perms(MLMData, nPerms)

tStats_2 = coeftable(GLMEst).cols[3]
pVals_2 = coeftable(GLMEst).cols[4]

@testset "mlmPermsTesting" begin
    # Testing the p value and t Statistics are similar between MatrixLM and GLM package
    @test isapprox(mean(tStats), mean(tStats_2), atol=20)
    @test isapprox(reshape(pVals,(200,1)), pVals_2, atol=tol)    
end
