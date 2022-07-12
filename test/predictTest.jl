using Test
using MatrixLM

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

fitted = MatrixLM.predict(MLMEst)
fitted2 = MatrixLM.fitted(MLMEst)

@test isapprox(MatrixLM.coef(MLMEst), B, atol=3)
@test isapprox(sum(fitted.Y - Y), 0, atol=1.3)

@test typeof(fitted) == typeof(fitted2)
@test isapprox(fitted.Y, fitted2.Y, atol=tol)

@test MatrixLM.calc_resid(get_X(MLMData), get_Y(MLMData), get_Z(MLMData),MatrixLM.coef(MLMEst)) == resid(MLMEst)
