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
# X: no intercept
# Z: no intercept
MLMData = RawData(Response(Y), Predictors(X, Z));

######################################
# Test: predicted and fitted results #
######################################

# Estimate Ŷpredict with X, Z intercept
MLMEst = mlm(MLMData, addXIntercept = true, addZIntercept = true) # WARNING IT CHANGES INTERCEPT IN ORGINAL MLMData
Ŷpredict_a = MatrixLM.predict(MLMEst).Y
Ŷfitted = MatrixLM.fitted(MLMEst).Y
residuals = resid(MLMEst, RawData(Response(Y),Predictors(X,Z,false,false)))
default_resids = resid(MLMEst)

@testset "predictTesting" begin   
    # testing the dimension of fitted y with actual Y, to see their consistancy
    @test sizeof(Ŷpredict_a) == sizeof(Y)
    @test sizeof(Ŷfitted) == sizeof(Y)
    @test isapprox(Ŷpredict_a, Ŷfitted, atol=tol)
    # testing the calc_preds function, too see if they are identical with the resid function
    @test MatrixLM.calc_resid(get_X(MLMEst.data), get_Y(MLMEst.data), get_Z(MLMEst.data),MatrixLM.coef(MLMEst)) == resid(MLMEst)
    @test default_resids == residuals    
end

###############################################
# Test: predicted results with new predictors #
###############################################

#MLMData2 = MLMData
# Testing new predicted estimation based on new predictors
# MLMEst.data.predictors.hasXIntercept -> true
# MLMEst.data.predictors.hasZIntercept -> true

# if previous X has X Intercept, and new X has not Intercept, it will add an intercept to new X
# if previous Z has Z Intercept, and new Z has not Intercept, it will add an intercept to new Z
Ŷpredict2 = MatrixLM.predict(MLMEst, Predictors(X,Z,false, false)).Y

# Estimate Ŷpredict without X, Z intercept
MLMEst = mlm(MLMData, addXIntercept = false, addZIntercept = false)
Ŷpredict_b = MatrixLM.predict(MLMEst).Y
default_resids = resid(MLMEst)
residuals2 = resid(MLMEst, RawData(Response(Y),Predictors(hcat(ones(size(X, 1)), X), hcat(ones(size(Z, 1)), Z),true, true)))
# MLMEst.data.predictors.hasXIntercept -> false
# MLMEst.data.predictors.hasZIntercept -> false

# if previous X has no X Intercept, and new X has Intercept, it will remove intercept from new X
# if previous Z has no Z Intercept, and new Z has Intercept, it will remove intercept from new Z
Ŷpredict3 = MatrixLM.predict(MLMEst, Predictors(hcat(ones(size(X, 1)), X), hcat(ones(size(Z, 1)), Z),true, true)).Y

@testset "resid_test" begin
    @test isapprox(sum(Ŷpredict_a - Ŷpredict2), 0, atol = 0.1 )
    @test isapprox(sum(Ŷpredict_b - Ŷpredict3), 0, atol = 0.1 )
    @test default_resids == residuals2
end
