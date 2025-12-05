
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

@testset "add_intercept" begin
    # Test with a matrix without intercept
    test_matrix = rand(10, 5)
    result = add_intercept(test_matrix)
    
    # Check dimensions - should have one more column
    @test size(result, 1) == size(test_matrix, 1)
    @test size(result, 2) == size(test_matrix, 2) + 1
    
    # Check that first column is all ones
    @test all(result[:, 1] .== 1)
    
    # Check that remaining columns match original matrix
    @test isapprox(result[:, 2:end], test_matrix, atol=tol)
    
    # Test with a matrix that already has intercept (first column of ones)
    matrix_with_intercept = hcat(ones(10), rand(10, 5))
    # This should print a warning but still work
    result_with_warning = add_intercept(matrix_with_intercept)
    @test size(result_with_warning, 2) == size(matrix_with_intercept, 2) + 1
    @test all(result_with_warning[:, 1] .== 1)
    
    # Test with edge case: single column matrix
    single_col = rand(10, 1)
    result_single = add_intercept(single_col)
    @test size(result_single, 2) == 2
    @test all(result_single[:, 1] .== 1)
    @test isapprox(result_single[:, 2], single_col[:, 1], atol=tol)
    
    # Test with edge case: single row matrix
    single_row = rand(1, 5)
    result_single_row = add_intercept(single_row)
    @test size(result_single_row, 1) == 1
    @test size(result_single_row, 2) == 6
    @test result_single_row[1, 1] == 1.0
    @test isapprox(result_single_row[:, 2:end], single_row, atol=tol)
end

@testset "remove_intercept" begin
    # Test with a matrix that has an intercept column
    test_matrix = rand(10, 5)
    matrix_with_intercept = hcat(ones(10), test_matrix)
    result = remove_intercept(matrix_with_intercept)
    
    # Check dimensions - should have one less column
    @test size(result, 1) == size(matrix_with_intercept, 1)
    @test size(result, 2) == size(matrix_with_intercept, 2) - 1
    
    # Check that the result matches the original matrix (without intercept)
    @test isapprox(result, test_matrix, atol=tol)
    
    # Test with a matrix without intercept (first column not all ones)
    # This should print a warning but still remove the first column
    matrix_no_intercept = rand(10, 5)
    result_with_warning = remove_intercept(matrix_no_intercept)
    @test size(result_with_warning, 2) == size(matrix_no_intercept, 2) - 1
    @test isapprox(result_with_warning, matrix_no_intercept[:, 2:end], atol=tol)
    
    # Test round-trip: add then remove intercept
    original = rand(15, 6)
    with_intercept = add_intercept(original)
    back_to_original = remove_intercept(with_intercept)
    @test isapprox(back_to_original, original, atol=tol)
    
    # Test with edge case: matrix with only intercept column
    only_intercept = ones(10, 1)
    result_only = remove_intercept(only_intercept)
    @test size(result_only, 1) == 10
    @test size(result_only, 2) == 0
    
    # Test with edge case: two-column matrix with intercept
    two_col = hcat(ones(8), rand(8))
    result_two_col = remove_intercept(two_col)
    @test size(result_two_col, 2) == 1
    @test isapprox(result_two_col[:, 1], two_col[:, 2], atol=tol)
    
    # Test that first column values are preserved when removing
    matrix_with_ones = hcat(ones(5), [1.0, 2.0, 3.0, 4.0, 5.0], rand(5, 2))
    removed = remove_intercept(matrix_with_ones)
    @test removed[1, 1] == 1.0
    @test removed[2, 1] == 2.0
    @test removed[3, 1] == 3.0
end

@testset "check_Z_rank" begin
    # Test with a full rank matrix - should not warn
    full_rank_matrix = rand(10, 5)
    # Use @test_nowarn to ensure no warning is issued
    @test_nowarn check_Z_rank(full_rank_matrix)
    
    # Test with a non-full rank matrix - should warn
    # Create a rank-deficient matrix by making one column a linear combination of others
    non_full_rank_matrix = rand(10, 5)
    non_full_rank_matrix[:, 3] = non_full_rank_matrix[:, 1] + non_full_rank_matrix[:, 2]
    # Use @test_logs to check that a warning is issued
    @test_logs (:warn, "The rank of Z matrix is not full, and this may generate errors.") check_Z_rank(non_full_rank_matrix)
    
    # Test with a matrix of duplicate columns (clearly not full rank)
    duplicate_cols = hcat(ones(8, 3), ones(8, 3))
    @test_logs (:warn, "The rank of Z matrix is not full, and this may generate errors.") check_Z_rank(duplicate_cols)
    
    # Test with a square full rank matrix
    square_full_rank = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
    @test_nowarn check_Z_rank(square_full_rank)
    
    # Test with a square singular matrix
    singular_matrix = [1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 9.0]
    @test_logs (:warn, "The rank of Z matrix is not full, and this may generate errors.") check_Z_rank(singular_matrix)

    # Test with a tall matrix (more rows than columns) that is full rank
    tall_matrix = rand(15, 5)
    @test_nowarn check_Z_rank(tall_matrix)
    
    # Test with a matrix where one column is all zeros (not full rank)
    matrix_with_zero_col = rand(10, 4)
    matrix_with_zero_col[:, 2] .= 0.0
    @test_logs (:warn, "The rank of Z matrix is not full, and this may generate errors.") check_Z_rank(matrix_with_zero_col)
end