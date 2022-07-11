using Test
using MatrixLM
using Random

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

@test shuffle_rows(X) != X
@test shuffle_cols(X) != X
@test isapprox(A_inter, hcat(ones(size(A,1)), A), atol=tol)
@test isapprox(remove_intercept(A_inter), A, atol=tol)
