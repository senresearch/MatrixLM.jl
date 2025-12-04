
###########################
# Generate Simulated Data #
###########################
n = 100
# Generate data with categorical variables and numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n),catvar3=rand(["D", "E"], n)), DataFrame(rand(n,4),:auto))

@testset "contr Function Tests" begin
    
    #####################################
    # Test 1: Basic contrast dimensions #
    #####################################
    @testset "Basic Dimensions" begin
        X1 = Matrix(contr(X_df, [:catvar1, :catvar2, :catvar3], ["treat", "sum", "noint"]))
        @test size(X1) == (100,12)
        
        X2 = Matrix(contr(X_df, [:catvar2, :catvar3], ["treat", "treat"], ["A","D"]))
        @test size(X2) == (100,8)
    end
    
    ############################################
    # Test 2: Treatment contrasts (no ref)    #
    ############################################
    @testset "Treatment Contrasts" begin
        df_small = DataFrame(cat=["A", "B", "C", "A", "B"], num=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = contr(df_small, [:cat], ["treat"])
        
        # Should have k-1 dummy columns for k levels
        @test size(result, 2) == 3  # 2 dummies + 1 numeric
        @test "cat_B" in names(result)
        @test "cat_C" in names(result)
        @test !("cat_A" in names(result))  # First level dropped
        
        # Check numeric correctness: B should be [0,1,0,0,1]
        @test result[2, :cat_B] == 1.0
        @test result[5, :cat_B] == 1.0
        @test result[1, :cat_B] == 0.0
        
        # Numeric column preserved
        @test result.num == df_small.num
    end
    
    ############################################
    # Test 3: Treatment contrasts with ref    #
    ############################################
    @testset "Treatment Contrasts with Reference" begin
        df_small = DataFrame(cat=["A", "B", "C", "A", "B"], num=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = contr(df_small, [:cat], ["treat"], ["B"])
        
        # B is reference, so A and C should be dummies
        @test size(result, 2) == 3
        @test "cat_A" in names(result)
        @test "cat_C" in names(result)
        @test !("cat_B" in names(result))
        
        # Check numeric correctness
        @test result[1, :cat_A] == 1.0
        @test result[4, :cat_A] == 1.0
        @test result[3, :cat_C] == 1.0
    end
    
    ############################################
    # Test 4: Sum contrasts                   #
    ############################################
    @testset "Sum Contrasts" begin
        df_small = DataFrame(cat=["A", "B", "C", "A", "B", "C"])
        result = contr(df_small, [:cat], ["sum"])
        
        # Should have k-1 columns, last level coded as -1
        @test size(result, 2) == 2
        @test "cat_A" in names(result)
        @test "cat_B" in names(result)
        
        # First level: [1, 0]
        @test result[1, :cat_A] == 1.0
        @test result[1, :cat_B] == 0.0
        
        # Second level: [0, 1]
        @test result[2, :cat_A] == 0.0
        @test result[2, :cat_B] == 1.0
        
        # Last level (C): [-1, -1]
        @test result[3, :cat_A] == -1.0
        @test result[3, :cat_B] == -1.0
    end
    
    ############################################
    # Test 5: No intercept contrasts          #
    ############################################
    @testset "No Intercept Contrasts" begin
        df_small = DataFrame(cat=["A", "B", "C", "A", "B"])
        result = contr(df_small, [:cat], ["noint"])
        
        # Should have k columns (all levels)
        @test size(result, 2) == 3
        @test "cat_A" in names(result)
        @test "cat_B" in names(result)
        @test "cat_C" in names(result)
        
        # Each row should have exactly one 1
        @test result[1, :cat_A] == 1.0
        @test result[2, :cat_B] == 1.0
        @test result[3, :cat_C] == 1.0
    end
    
    ############################################
    # Test 6: Sum no intercept contrasts      #
    ############################################
    @testset "Sum No Intercept Contrasts" begin
        df_small = DataFrame(cat=["A", "B", "C"])
        result = contr(df_small, [:cat], ["sumnoint"])
        
        # Should have k columns
        @test size(result, 2) == 3
        
        # Each column should sum to 0 (centered)
        # Values should be 1 - 1/k for own level, -1/k for others
        k = 3
        expected_own = 1.0 - 1.0/k
        expected_other = -1.0/k
        
        @test result[1, :cat_A] ≈ expected_own
        @test result[1, :cat_B] ≈ expected_other
        @test result[1, :cat_C] ≈ expected_other
        @test sum(result[1, :]) ≈ 0.0 atol=1e-10
    end
    
    ############################################
    # Test 7: Multiple variables mixed types  #
    ############################################
    @testset "Multiple Variables" begin
        df = DataFrame(
            cat1=["A", "B", "A", "B"],
            cat2=["X", "Y", "Z", "X"],
            num=[1.0, 2.0, 3.0, 4.0]
        )
        result = contr(df, [:cat1, :cat2], ["treat", "sum"])
        
        # cat1: 1 dummy (B), cat2: 2 dummies (X, Y), num: 1
        @test size(result, 2) == 4
        @test "cat1_B" in names(result)
        @test "cat2_X" in names(result)
        @test "cat2_Y" in names(result)
        @test "num" in names(result)
    end
    
    ############################################
    # Test 8: Edge case - empty Symbol        #
    ############################################
    @testset "Empty Symbol Edge Case" begin
        df_orig = DataFrame(num1=[1.0, 2.0], num2=[3.0, 4.0])
        result = contr(df_orig, [Symbol()], ["treat"])
        
        # Should return original DataFrame unchanged
        @test result == df_orig
        @test names(result) == names(df_orig)
    end
    
    ############################################
    # Test 9: Preserve column order           #
    ############################################
    @testset "Column Order" begin
        df = DataFrame(num1=[1.0, 2.0], cat=["A", "B"], num2=[3.0, 4.0])
        result = contr(df, [:cat], ["treat"])
        
        # Non-categorical columns should appear in original order
        # num1 first, then cat dummy, then num2
        @test names(result)[1] == "num1"
        @test names(result)[end] == "num2"
    end
    
    ############################################
    # Test 10: Error handling                 #
    ############################################
    @testset "Error Handling" begin
        df = DataFrame(cat=["A", "B", "C"])
        
        # Invalid contrast type
        @test_throws ErrorException contr(df, [:cat], ["invalid"])
        
        # Variable not in DataFrame
        @test_throws ErrorException contr(df, [:nonexistent], ["treat"])
        
        # trtRef with not existing level
        @test_throws ErrorException contr(df, [:cat], ["sum"], ["D"])
    end
    
    ############################################
    # Test 11: Numeric types preserved        #
    ############################################
    @testset "Numeric Type Preservation" begin
        df = DataFrame(cat=["A", "B"], int_col=Int64[1, 2], float_col=[1.5, 2.5])
        result = contr(df, [:cat], ["treat"])
        
        # Original numeric columns should maintain their types
        @test eltype(result.int_col) == Int64
        @test eltype(result.float_col) == Float64
    end
    
    ############################################
    # Test 12: Single-level category         #
    ############################################
    @testset "Single Level Category" begin
        df = DataFrame(cat=["A", "A", "A"])
        
        # Treatment: should produce 0 columns (k-1 = 0)
        result_treat = contr(df, [:cat], ["treat"])
        @test size(result_treat, 2) == 0
        
        # No intercept: should produce 1 column
        result_noint = contr(df, [:cat], ["noint"])
        @test size(result_noint, 2) == 1
        @test all(result_noint[:, 1] .== 1.0)
    end

end
