
###########################
# Generate Simulated Data #
###########################
n = 100
# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n),catvar3=rand(["D", "E"], n)), DataFrame(rand(n,4),:auto))

X1 = Matrix(contr(X_df, [:catvar1, :catvar2, :catvar3], ["treat", "sum", "noint"]))

#MatrixLM.get_dummy(X_df,:catvar2,"treat","A")

X2 = Matrix(contr(X_df, [:catvar2, :catvar3], ["treat", "treat"], ["A","D"]))

@testset "contrTest" begin
    @test size(X1) == (100,12)
    @test size(X2) == (100,8)
end

@testset "get_dummy_treatment_contrasts" begin
    # Create a small test DataFrame
    test_df = DataFrame(category = ["A", "B", "C", "A", "B", "C"])
    
    # Test treatment contrasts with default reference (last level when unspecified)
    result = MatrixLM.get_dummy(test_df, :category, "treat")
    
    # Check dimensions: should have n-1 columns for n categories
    @test size(result, 1) == 6
    @test size(result, 2) == 2  # 3 categories - 1 reference
    
    # Check column names
    @test names(result) == ["category_A", "category_B"]
    
    # Check values - C is reference (all zeros), A and B get indicator columns
    @test result[1, :category_A] == 1.0  # First row is A
    @test result[1, :category_B] == 0.0
    @test result[2, :category_A] == 0.0  # Second row is B
    @test result[2, :category_B] == 1.0
    @test result[3, :category_A] == 0.0  # Third row is C (reference)
    @test result[3, :category_B] == 0.0
    
    # Providing an empty string should behave like omitting the reference
    result_empty = MatrixLM.get_dummy(test_df, :category, "treat", "")
    @test names(result_empty) == names(result)
    @test Matrix(result_empty) == Matrix(result)
    
    # Test treatment contrasts with specified reference
    result_ref = MatrixLM.get_dummy(test_df, :category, "treat", "B")
    
    # Check dimensions
    @test size(result_ref, 1) == 6
    @test size(result_ref, 2) == 2
    
    # Check column names - B should not be included
    @test names(result_ref) == ["category_A", "category_C"]
    
    # Check values - B is reference (all zeros)
    @test result_ref[1, :category_A] == 1.0  # First row is A
    @test result_ref[1, :category_C] == 0.0
    @test result_ref[2, :category_A] == 0.0  # Second row is B (reference)
    @test result_ref[2, :category_C] == 0.0
    @test result_ref[3, :category_A] == 0.0  # Third row is C
    @test result_ref[3, :category_C] == 1.0
end

@testset "get_dummy_sum_contrasts" begin
    # Create a test DataFrame
    test_df = DataFrame(category = ["A", "B", "C", "D", "A", "B"])
    
    # Test sum contrasts with default reference (first sorted level when unspecified)
    result = MatrixLM.get_dummy(test_df, :category, "sum")
    
    # Check dimensions: should have n-1 columns for n categories
    @test size(result, 1) == 6
    @test size(result, 2) == 3  # 4 categories - 1
    
    # Check column names (should skip first level and exclude last)
    @test names(result) == ["category_B", "category_C", "category_D"]
    
    # Check values for sum contrasts
    # Reference level (A) should get -1 for all its dummy columns
    @test result[1, :category_B] == -1.0  # Row with A
    @test result[1, :category_C] == -1.0
    @test result[1, :category_D] == -1.0
    
    # Other categories get standard indicators
    @test result[2, :category_B] == 1.0  # Row with B
    @test result[2, :category_C] == 0.0
    @test result[2, :category_D] == 0.0
    
    @test result[3, :category_B] == 0.0  # Row with C
    @test result[3, :category_C] == 1.0
    @test result[3, :category_D] == 0.0
    
    # Last category (D) receives standard indicator
    @test result[4, :category_B] == 0.0  # Row with D
    @test result[4, :category_C] == 0.0
    @test result[4, :category_D] == 1.0

    # Sum contrasts with an explicit reference level (use C)
    result_ref = MatrixLM.get_dummy(test_df, :category, "sum", "C")
    @test size(result_ref) == (6, 3)
    @test names(result_ref) == ["category_A", "category_B", "category_D"]

    # Rows corresponding to reference level (C) should be -1 across columns
    idx_C = findfirst(==("C"), test_df.category)
    @test all(Matrix(result_ref)[idx_C, :] .== -1.0)

    # Non-reference rows retain indicator coding
    idx_A = findfirst(==("A"), test_df.category)
    @test result_ref[idx_A, :category_A] == 1.0
    @test result_ref[idx_A, :category_B] == 0.0
    @test result_ref[idx_A, :category_D] == 0.0

    idx_B = findfirst(==("B"), test_df.category)
    @test result_ref[idx_B, :category_A] == 0.0
    @test result_ref[idx_B, :category_B] == 1.0
    @test result_ref[idx_B, :category_D] == 0.0
end

@testset "get_dummy_noint_contrasts" begin
    # Create a test DataFrame
    test_df = DataFrame(category = ["X", "Y", "Z", "X", "Y"])
    
    # Test no-intercept contrasts
    result = MatrixLM.get_dummy(test_df, :category, "noint")
    
    # Check dimensions: should have n columns for n categories (all levels included)
    @test size(result, 1) == 5
    @test size(result, 2) == 3  # All 3 categories included
    
    # Check column names
    @test names(result) == ["category_X", "category_Y", "category_Z"]
    
    # Check values - standard indicator coding for all levels
    @test result[1, :category_X] == 1.0  # First row is X
    @test result[1, :category_Y] == 0.0
    @test result[1, :category_Z] == 0.0
    
    @test result[2, :category_X] == 0.0  # Second row is Y
    @test result[2, :category_Y] == 1.0
    @test result[2, :category_Z] == 0.0
    
    @test result[3, :category_X] == 0.0  # Third row is Z
    @test result[3, :category_Y] == 0.0
    @test result[3, :category_Z] == 1.0
end

@testset "get_dummy_sumnoint_contrasts" begin
    # Create a test DataFrame with 3 categories
    test_df = DataFrame(category = ["A", "B", "C", "A", "B", "C"])
    
    # Test sum-no-intercept contrasts
    result = MatrixLM.get_dummy(test_df, :category, "sumnoint")
    
    # Check dimensions: should have n columns for n categories
    @test size(result, 1) == 6
    @test size(result, 2) == 3
    
    # Check column names
    @test names(result) == ["category_A", "category_B", "category_C"]
    
    # Check values - each category gets (1 - 1/n_categories) when present
    # and (-1/n_categories) when absent
    n_categories = 3
    expected_present = 1.0 - (1.0 / n_categories)
    expected_absent = -(1.0 / n_categories)
    
    # First row is A
    @test isapprox(result[1, :category_A], expected_present, atol=1e-10)
    @test isapprox(result[1, :category_B], expected_absent, atol=1e-10)
    @test isapprox(result[1, :category_C], expected_absent, atol=1e-10)
    
    # Second row is B
    @test isapprox(result[2, :category_A], expected_absent, atol=1e-10)
    @test isapprox(result[2, :category_B], expected_present, atol=1e-10)
    @test isapprox(result[2, :category_C], expected_absent, atol=1e-10)
    
    # Each column should sum to approximately 0 (sum contrast property)
    @test isapprox(sum(result[:, :category_A]), 0.0, atol=1e-10)
    @test isapprox(sum(result[:, :category_B]), 0.0, atol=1e-10)
    @test isapprox(sum(result[:, :category_C]), 0.0, atol=1e-10)
end

@testset "get_dummy_edge_cases" begin
    # Test with binary variable
    binary_df = DataFrame(binary = ["Yes", "No", "Yes", "No"])
    result_binary = MatrixLM.get_dummy(binary_df, :binary, "treat")
    @test size(result_binary, 2) == 1  # Binary -> 1 dummy
    
    # Test with single level (edge case)
    single_df = DataFrame(single = ["A", "A", "A"])
    result_single_noint = MatrixLM.get_dummy(single_df, :single, "noint")
    @test size(result_single_noint, 2) == 1  # Single level with noint
    @test all(result_single_noint[:, 1] .== 1.0)
    
    # Test with many levels
    many_levels_df = DataFrame(many = string.(1:10))
    result_many = MatrixLM.get_dummy(many_levels_df, :many, "treat")
    @test size(result_many, 2) == 9  # 10 levels - 1 reference
    
    # Test error handling for invalid contrast type
    test_df = DataFrame(cat = ["A", "B", "C"])
    @test_throws ErrorException MatrixLM.get_dummy(test_df, :cat, "invalid")
    
    # Test error handling for trtRef with unsupported contrast
    @test_throws ErrorException MatrixLM.get_dummy(test_df, :cat, "noint", "A")

    # Invalid reference level should throw for treat and sum
    @test_throws ErrorException MatrixLM.get_dummy(test_df, :cat, "treat", "Z")
    @test_throws ErrorException MatrixLM.get_dummy(test_df, :cat, "sum", "Z")
end

@testset "get_dummy_numeric_categories" begin
    # Test with numeric categories (converted to strings internally)
    numeric_df = DataFrame(num_cat = [1, 2, 3, 1, 2, 3])
    result = MatrixLM.get_dummy(numeric_df, :num_cat, "treat")
    
    @test size(result, 1) == 6
    @test size(result, 2) == 2  # 3 categories - 1 reference
    
    # Column names should include the numeric values as strings
    @test "num_cat_1" in names(result)
    @test "num_cat_2" in names(result)
end
