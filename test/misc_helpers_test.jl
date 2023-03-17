###########
# Library #
###########
using Test
using MatrixLM
using Random


###########################
# Generate Simulated Data #
###########################

# Tolerance for tests
tol = 10.0^(-7)
    
# Dimensions of matrices 
n = 100
m = 200
p = 10
    
# Generate some matrices.
Random.seed!(1)
X = rand(n,p)


A = zeros(Float64, n, m)
A_inter = copy(add_intercept(A))


@testset "miscHelperTesting" begin
    # Testing the shuffle function
    @test isapprox(sum(shuffle_rows(X),dims=1), sum(X,dims=1))
    @test isapprox(sum(shuffle_cols(X),dims=2), sum(X,dims=2))
    # Testing remove_intercept and add_intercept
    @test isapprox(A_inter, hcat(ones(size(A,1)), A), atol=tol)
    @test isapprox(remove_intercept(A_inter), A, atol=tol)    
end