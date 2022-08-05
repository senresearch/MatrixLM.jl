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
tol = 10.0^(-7)
    
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
w = rand(Float64, m)
W = diagm(w)
WZ = W * Z
Yw = X*B*transpose(WZ)+E
    
# Data frame to be passed into lm
GLMData = DataFrame(hcat(vec(Y), kron(Z,X)), :auto)
# lm estimate
GLMEst = lm(Matrix(GLMData[:,2:end]), Vector(GLMData[:,1]))
    
# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
MLMData_w = RawData(Response(Yw), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, addXIntercept = false, addZIntercept = false)
MLMEst = mlm(MLMData, addXIntercept = false, addZIntercept = false)
    
@testset "testmlm" begin
    @test isapprox(GLM.coef(GLMEst), vec(MatrixLM.coef(MLMEst)), atol=tol)
    @test isapprox(GLM.predict(GLMEst), vec(MatrixLM.predict(MLMEst).Y), atol=tol)
    #@test LinearAlgebra.issymmetric(round.(MLMEst.sigma, digits=10))
end


MLMEst_w = mlm(MLMData_w, weights = w , addXIntercept = true, addZIntercept = false, targetType = 'E')
GLMData_w = DataFrame(hcat(vec(Yw), kron(WZ,X)), :auto)
GLMEst_w = lm(Matrix(GLMData_w[:,2:end]), Vector(GLMData_w[:,1]))

@testset "weightedMlmTest" begin
    @test isapprox(GLM.coef(GLMEst_w), vec(MatrixLM.calc_coeffs(X,Yw,W*Z,transpose(X)*X,transpose(Z)*W*W*Z)), atol=tol)
    #@test isapprox(GLM.predict(GLMEst_w), vec(MatrixLM.predict(MLMEst_w).Y), atol=100000)
    @test LinearAlgebra.issymmetric(round.(MLMEst_w.sigma, digits=10))
    #@test isapprox(GLM.coef(GLMEst_w), vec(MatrixLM.coef(MLMEst_w)), atol=tol)
end;
