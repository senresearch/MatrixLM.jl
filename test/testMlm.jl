using Test
using MatrixLM

using DataFrames
using Random
using LinearAlgebra
using GLM

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
    
# Data frame to be passed into lm
GLMData = DataFrame(hcat(vec(Y), kron(Z,X)), :auto)
# lm estimate
GLMEst = lm(Matrix(GLMData[:,2:end]), Vector(GLMData[:,1]))
    
# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)
MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)
    
@test isapprox(GLM.coef(GLMEst), vec(MatrixLM.coef(MLMEst)), atol=tol)
@test isapprox(GLM.predict(GLMEst), vec(MatrixLM.predict(MLMEst).Y), atol=tol)
@test LinearAlgebra.issymmetric(round.(MLMEst.sigma, digits=10)) # and also positive semi-definite?

