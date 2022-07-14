###########
# Library #
###########
using Test
using MatrixLM


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

# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)
MLMEst = mlm(MLMData, hasXIntercept = false, hasZIntercept = false)

fitted = MatrixLM.predict(MLMEst)
fitted2 = MatrixLM.fitted(MLMEst)

# Testing the coefficients of the model with simulated B Matrix
@test isapprox(MatrixLM.coef(MLMEst), B, atol=3)
# testing the dimension of fitted y with actual Y, to see their consistancy
@test sizeof(fitted.Y) == sizeof(Y)
@test sizeof(fitted2.Y) == sizeof(Y)
@test typeof(fitted) == typeof(fitted2)
@test isapprox(fitted.Y, fitted2.Y, atol=tol)
# testing the calc_preds function, too see if they are identical with the resid function
@test MatrixLM.calc_resid(get_X(MLMData), get_Y(MLMData), get_Z(MLMData),MatrixLM.coef(MLMEst)) == resid(MLMEst)

# testing the model with intercept
resid_1 = resid(MLMEst)
MLMEst_inter = mlm(MLMData, hasXIntercept = true, hasZIntercept = true)
resid_inter = MatrixLM.resid(MLMEst_inter)
@test size(resid_1) == size(resid_inter)