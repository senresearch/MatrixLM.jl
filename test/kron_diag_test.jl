
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


@testset "diagonalTesting" begin
    # 2d matrix -> returns diagonal vector
    A = Float64[1 2 3; 4 5 6; 7 8 9]
    dA = MatrixLM.diagonal(A)
    @test dA == Float64[1,5,9]

    # 1d vector -> returns itself
    v = Float64[1,2,3]
    dv = MatrixLM.diagonal(v)
    @test dv === v

    # scalar -> returns itself
    s = 2.5
    ds = MatrixLM.diagonal(s)
    @test ds == s
end