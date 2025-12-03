
# Tests for calc_sigma

# Tolerance for tests
tol = 10.0^(-7)

# Small deterministic example
Random.seed!(1)
n = 5
m = 3
rsdl = randn(n,m)  

@testset "calcSigmaBasic" begin
    # When targetType is nothing, should compute sample covariance
    # Sigma = RSS / (n - 1)
    RSS = transpose(rsdl) * rsdl
    expected_sigma = RSS ./ (n - 1)

    sigma, lambda = MatrixLM.calc_sigma(rsdl, nothing)

    @test size(sigma) == (m, m)
    @test isapprox(sigma, expected_sigma, atol=tol)
    @test lambda == 0.0
end

@testset "calcSigmaDelegation" begin
    # When targetType is a string, calc_sigma should delegate to shrink_sigma
    s1, l1 = MatrixLM.calc_sigma(rsdl, "A")
    s2, l2 = MatrixLM.shrink_sigma(rsdl, "A")

    @test size(s1) == (m, m)
    @test size(s2) == (m, m)
    @test isapprox(s1, s2, atol=tol)
    @test isapprox(l1, l2, atol=tol)
    @test 0.0 <= l1 <= 1.0
end
