###########
# Library #
###########
using MatrixLM, Random, Test

###########################
# Generate Simulated Data #
###########################
n = 100
p = 10
X = rand(n,n)
Z = rand(p,p)


X_vec = rand(n)
mat = MatrixLM.kron_diag(X,Z)
mat2 = MatrixLM.kron_diag(X_vec,Z)

@testset "kronTesting" begin
    # Test the dimensions after the transformation
    @test size(mat) == size(mat2) == (p,n)
end