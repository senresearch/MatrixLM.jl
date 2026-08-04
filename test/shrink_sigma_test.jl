
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
rng = StableRNG(4)
X = rand(rng, n, p)
    
X_mean = mean(X, dims=1)
est, varEst = MatrixLM.cov_est(X)
lambda = MatrixLM.shrink_sigma(X, "A")[2]
T = Matrix{Float64}(I, p, p)
lambda2 = sum(varEst) / sum((est-T).^2)
lambdaB = MatrixLM.shrink_sigma(X, "B")[2]
lambdaC = MatrixLM.shrink_sigma(X, "C")[2]
lambdaD = MatrixLM.shrink_sigma(X, "D")[2]

@testset "shrinkSigmaTesting" begin
    @test isapprox(lambda, lambda2, atol=tol)
    # compare the center function
    @test isapprox(mean(MatrixLM.center(X)), 0, atol=tol)
    # compare the result of cov_est with cov function
    @test isapprox(est, cov(X), atol=tol)
    # test the size of variance matrices
    @test size(varEst) == (p,p)
    # Test 1. compare shrink_sigma function
    @test isapprox(lambdaB, lambdaC, atol=0.1)
    @test isapprox(lambdaB, lambdaD, atol=0.1)
    @test isapprox(lambdaC, lambdaD, atol= 0.1)
end;

@testset "tri2vec and vec2tri checks" begin
        A = [1.0 0.2 0.3;
             0.2 2.0 0.4;
             0.3 0.4 3.0]

    x, d = MatrixLM.tri2vec(A)
    A_reconstructed = MatrixLM.vec2tri(x, d)
    @test isapprox(A_reconstructed, A, atol=tol)
end;

@testset "tri2vec rejects non-square matrices" begin
    rng = StableRNG(41)
    A = randn(rng, 3, 4)
    @test_throws ErrorException MatrixLM.tri2vec(A)

end;

@testset "vec2tri rejects incompatible dimensions" begin
    x = [0.1, 0.2, 0.3]   # corresponds to off-diagonal entries of a 3x3 matrix
    d = [1.0, 2.0]        # wrong diagonal length
    @test_throws ErrorException MatrixLM.vec2tri(x, d)
end;

@testset "shrink_var returns positive definite covariance matrix" begin

    rng = StableRNG(123)

    # n < p to make the sample covariance singular/noisy
    X_hd = randn(rng, 10, 25)

    S, lambdaR = MatrixLM.shrink_var(X_hd)

    @test size(S) == (25, 25)
    @test isapprox(S, S', atol=tol)
    @test isposdef(Symmetric(S))

    @test lambdaR isa NamedTuple
    @test hasproperty(lambdaR, :correlation)
    @test hasproperty(lambdaR, :variance)

    @test 0.0 <= lambdaR.correlation <= 1.0
    @test 0.0 <= lambdaR.variance <= 1.0
end;

@testset "shrink_sigma target R routing" begin

    rng = StableRNG(124)
    X_R = randn(rng, 20, 12)

    S_direct, lambda_direct = MatrixLM.shrink_var(X_R)
    S_routed, lambda_routed = MatrixLM.shrink_sigma(X_R, "R")

    @test isapprox(S_routed, S_direct, atol=tol)

    @test isapprox(
        lambda_routed.correlation,
        lambda_direct.correlation,
        atol=tol
    )

    @test isapprox(
        lambda_routed.variance,
        lambda_direct.variance,
        atol=tol
    )

    @test isposdef(S_routed)
end;

@testset "shrink_sigma rejects invalid target" begin

    rng = StableRNG(125)
    X_invalid = randn(rng, 20, 5)

    @test_throws ArgumentError MatrixLM.shrink_sigma(
        X_invalid,
        "invalid"
    )
end;

@testset "shrink_var rejects sample size <= 3" begin

    rng = StableRNG(126)
    X_small = randn(rng, 3, 5)

    @test_throws ErrorException MatrixLM.shrink_var(X_small)
end;