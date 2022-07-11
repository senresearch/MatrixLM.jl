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
    
# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)
MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)

nPerms = 5
tStats, pVals = mlm_perms(MLMData, nPerms)

function mlm_t_stat(data::RawData)
    return t_stat(mlm(data; hasXIntercept=true, 
                  hasZIntercept=true, weights=nothing, 
                  targetType=nothing))
end

tStats_2, pVals_2 = perm_pvals(mlm_t_stat, MLMData, nPerms; permFun=shuffle_rows)
@test isapprox(tStats, tStats_2, atol=tol)
@test isapprox(pVals, pVals_2, atol=tol)