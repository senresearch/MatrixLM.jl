
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

@testset "diagonal_scalar" begin
    # Test with a positive scalar
    scalar1 = 5.0
    result1 = MatrixLM.diagonal(scalar1)
    @test result1 === scalar1
    @test result1 == 5.0
    @test typeof(result1) == Float64
    
    # Test with a negative scalar
    scalar2 = -3.14
    result2 = MatrixLM.diagonal(scalar2)
    @test result2 === scalar2
    @test result2 == -3.14
    
    # Test with zero
    scalar3 = 0.0
    result3 = MatrixLM.diagonal(scalar3)
    @test result3 === scalar3
    @test result3 == 0.0
    
    # Test with a very small scalar
    scalar4 = 1e-10
    result4 = MatrixLM.diagonal(scalar4)
    @test result4 === scalar4
    @test result4 == 1e-10
    
    # Test with a very large scalar
    scalar5 = 1e10
    result5 = MatrixLM.diagonal(scalar5)
    @test result5 === scalar5
    @test result5 == 1e10
end

@testset "diagonal_1d_array" begin
    # Test with a 1d array
    vec1 = [1.0, 2.0, 3.0]
    result1 = MatrixLM.diagonal(vec1)
    @test result1 === vec1
    @test result1 == [1.0, 2.0, 3.0]
    
    # Test with a single-element array
    vec2 = [5.0]
    result2 = MatrixLM.diagonal(vec2)
    @test result2 === vec2
    @test result2 == [5.0]
    
    # Test with a longer array
    vec3 = rand(10)
    result3 = MatrixLM.diagonal(vec3)
    @test result3 === vec3
    @test all(result3 .== vec3)
end

@testset "diagonal_2d_array" begin
    # Test with a square matrix
    mat1 = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    result1 = MatrixLM.diagonal(mat1)
    @test result1 == [1.0, 5.0, 9.0]
    @test length(result1) == 3
    
    # Test with a 2x2 matrix
    mat2 = [1.0 2.0; 3.0 4.0]
    result2 = MatrixLM.diagonal(mat2)
    @test result2 == [1.0, 4.0]
    
    # Test with a 1x1 matrix
    mat3 = reshape([7.0], 1, 1)
    result3 = MatrixLM.diagonal(mat3)
    @test result3 == [7.0]
    @test length(result3) == 1
    
    # Test with a rectangular matrix (more rows than columns)
    mat4 = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    result4 = MatrixLM.diagonal(mat4)
    @test result4 == [1.0, 4.0]
    
    # Test with a rectangular matrix (more columns than rows)
    mat5 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    result5 = MatrixLM.diagonal(mat5)
    @test result5 == [1.0, 5.0]
end