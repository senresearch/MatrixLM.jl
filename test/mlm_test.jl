
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
    
# Dataframe to be passed into lm
GLMData = DataFrame(hcat(vec(Y), kron(Z,X)), :auto)
# lm estimate
GLMEst = lm(Matrix(GLMData[:,2:end]), Vector(GLMData[:,1]))
    
# Put together RawData object for MLM
MLMData = RawData(Response(Y), Predictors(X, Z))
MLMData_w = RawData(Response(Yw), Predictors(X, Z))
# mlm estimate
# MLMEst = mlm(MLMData, addXIntercept = false, addZIntercept = false)
MLMEst = mlm(MLMData, addXIntercept = false, addZIntercept = false)

# Function to run matrix linear model on a fresh copy of the data
fresh_raw() = RawData(Response(copy(Y)), Predictors(copy(X), copy(Z)))
    
@testset "testmlm" begin
    @test isapprox(GLM.coef(GLMEst), vec(MatrixLM.coef(MLMEst)), atol=tol)
    @test isapprox(GLM.predict(GLMEst), vec(MatrixLM.predict(MLMEst).Y), atol=tol)
    #@test LinearAlgebra.issymmetric(round.(MLMEst.sigma, digits=10))
end

@testset "mlm targetType R integration" begin
    MLMEst_R = mlm(
        fresh_raw();
        addXIntercept = false,
        addZIntercept = false,
        targetType = "R"
    )

    # Confirm that the requested target type is retained
    @test MLMEst_R.targetType == "R"

    # The residual covariance should have one row and column
    # for each response variable
    @test size(MLMEst_R.sigma) == (m, m)

    # The covariance estimate should be symmetric and positive definite
    @test isapprox(MLMEst_R.sigma, MLMEst_R.sigma', atol=tol)
    @test isposdef(Symmetric(MLMEst_R.sigma))

    # The R estimator should return two named shrinkage coefficients
    @test MLMEst_R.lambda isa NamedTuple
    @test hasproperty(MLMEst_R.lambda, :correlation)
    @test hasproperty(MLMEst_R.lambda, :variance)

    # Both shrinkage coefficients should lie between zero and one
    @test 0.0 <= MLMEst_R.lambda.correlation <= 1.0
    @test 0.0 <= MLMEst_R.lambda.variance <= 1.0

    # Confirm that ordinary model outputs are still produced
    @test size(MLMEst_R.B) == (p, q)
    @test size(MLMEst_R.varB) == (p, q)

end

@testset "mlm Boolean variance shrinkage interface" begin
    # varShrinkage = false should be equivalent to targetType = nothing
    MLMEst_false = mlm(
        fresh_raw(),
        false;
        addXIntercept = false,
        addZIntercept = false
    )

    MLMEst_none = mlm(
        fresh_raw();
        addXIntercept = false,
        addZIntercept = false,
        targetType = nothing
    )

    @test MLMEst_false.targetType === nothing
    @test MLMEst_none.targetType === nothing

    @test isapprox(MLMEst_false.B, MLMEst_none.B, atol=tol)
    @test isapprox(MLMEst_false.sigma, MLMEst_none.sigma, atol=tol)
    @test isapprox(MLMEst_false.varB, MLMEst_none.varB, atol=tol)

    # varShrinkage = true should be equivalent to targetType = "R"
    MLMEst_true = mlm(
        fresh_raw(),
        true;
        addXIntercept = false,
        addZIntercept = false
    )

    MLMEst_R = mlm(
        fresh_raw();
        addXIntercept = false,
        addZIntercept = false,
        targetType = "R"
    )

    @test MLMEst_true.targetType == "R"
    @test MLMEst_R.targetType == "R"

    @test isapprox(MLMEst_true.B, MLMEst_R.B, atol=tol)
    @test isapprox(MLMEst_true.sigma, MLMEst_R.sigma, atol=tol)
    @test isapprox(MLMEst_true.varB, MLMEst_R.varB, atol=tol)
    @test MLMEst_true.lambda == MLMEst_R.lambda
end


MLMEst_w = mlm(MLMData_w, weights = w , addXIntercept = true, addZIntercept = false, targetType = "E")
GLMData_w = DataFrame(hcat(vec(Yw), kron(WZ,X)), :auto)
GLMEst_w = lm(Matrix(GLMData_w[:,2:end]), Vector(GLMData_w[:,1]))

@testset "weightedMlmTest" begin
    @test isapprox(GLM.coef(GLMEst_w), vec(MatrixLM.calc_coeffs(X,Yw,W*Z,transpose(X)*X,transpose(Z)*W*W*Z)), atol=tol)
    #@test isapprox(GLM.predict(GLMEst_w), vec(MatrixLM.predict(MLMEst_w).Y), atol=100000)
    @test LinearAlgebra.issymmetric(round.(MLMEst_w.sigma, digits=10))
    #@test isapprox(GLM.coef(GLMEst_w), vec(MatrixLM.coef(MLMEst_w)), atol=tol)
end;

@testset "weighted mlm rejects invalid weight length" begin
    bad_weights = rand(Float64, m - 1)

    @test_throws ErrorException mlm(
        fresh_raw();
        weights = bad_weights,
        addXIntercept = false,
        addZIntercept = false
    )
end;


@testset "tStatTest" begin
    # test t statistics with and without main effects, and with only one intercept
    MLM_intercepts = mlm(fresh_raw(), addXIntercept = true, addZIntercept = true)
    t_no_main = MatrixLM.t_stat(MLM_intercepts)
    expected_no_main = MLM_intercepts.B[2:end, 2:end] ./ sqrt.(MLM_intercepts.varB[2:end, 2:end])
    @test size(t_no_main) == size(expected_no_main)
    @test t_no_main ≈ expected_no_main atol = tol

    # test with main effects included
    t_with_main = MatrixLM.t_stat(MLM_intercepts, true)
    expected_with_main = MLM_intercepts.B ./ sqrt.(MLM_intercepts.varB)
    @test t_with_main ≈ expected_with_main atol = tol

    # test with only one intercept included
    MLM_x_only = mlm(fresh_raw(), addXIntercept = true, addZIntercept = false)
    t_x_only = MatrixLM.t_stat(MLM_x_only)
    expected_x_only = MLM_x_only.B[2:end, :] ./ sqrt.(MLM_x_only.varB[2:end, :])
    @test t_x_only ≈ expected_x_only atol = tol

    # test with only Z intercept included
    MLM_z_only = mlm(fresh_raw(), addXIntercept = false, addZIntercept = true)
    t_z_only = MatrixLM.t_stat(MLM_z_only)
    expected_z_only = MLM_z_only.B[:, 2:end] ./ sqrt.(MLM_z_only.varB[:, 2:end])
    @test t_z_only ≈ expected_z_only atol = tol
end
