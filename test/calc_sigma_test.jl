
# Tests for calc_sigma

# Tolerance for tests
tol = 10.0^(-7)

# Small deterministic example
Random.seed!(1)
n = 5
m = 3
resid = randn(n,m)

@testset "calcSigmaBasic" begin
    # targetType = nothing: manual RSS/(n-1)
    RSS = transpose(resid) * resid
    expected_sigma = RSS ./ (n - 1)

    sigma, lambda = MatrixLM.calc_sigma(resid, nothing)

    @test size(sigma) == (m, m)
    @test isapprox(sigma, expected_sigma, atol=tol)
    @test lambda == 0.0
end

@testset "calcSigmaDelegation" begin
    # when targetType is a string, calc_sigma should delegate to shrink_sigma
    s1, l1 = MatrixLM.calc_sigma(resid, "A")
    s2, l2 = MatrixLM.shrink_sigma(resid, "A")

    @test size(s1) == (m, m)
    @test size(s2) == (m, m)
    @test isapprox(s1, s2, atol=tol)
    @test isapprox(l1, l2, atol=tol)
    @test 0.0 <= l1 <= 1.0
end
