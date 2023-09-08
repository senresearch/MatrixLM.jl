
###########################
# Generate Simulated Data #
###########################
n = 100
# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(
    DataFrame(
        catvar1=string.(rand(0:1, n)), 
        catvar2=rand(["A", "B", "C", "D"], n),
        catvar3=rand([ "D", "E"], n)
    ), 
    DataFrame(rand(n,3), ["var3", "var4", "var5"])
    );


my_contrasts = Dict(
            :catvar1 => DummyCoding(), 
            :catvar2 => EffectsCoding(base = "A"),
            :catvar3 =>DummyCoding()
            )

mat = my_contrasts = Dict(
    :catvar1 => DummyCoding(), 
    :catvar2 => EffectsCoding(base = "A"),
    :catvar3 =>DummyCoding()
    )
    
mat2 = MatrixLM.design_matrix(
        @mlmformula(1 + catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4), 
        X_df, 
        my_contrasts
       )
mat3 = MatrixLM.design_matrix(
        @mlmformula(1 + catvar1 + catvar2), 
        X_df
       )

mat_terms = MatrixLM.design_matrix_names(
        @mlmformula(1 + catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4 ),
        X_df,
        [(:catvar1, :catvar3, DummyCoding()) , (:catvar2, EffectsCoding()) ]  
      )
mat2_terms = MatrixLM.design_matrix_names(
        @mlmformula(1 + catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4), 
        X_df, 
        my_contrasts
       )
mat3_terms = MatrixLM.design_matrix_names(
        @mlmformula(1 + catvar1 + catvar2), 
        X_df
       )




@testset "designMatrixTesting" begin
    # test the dimension of the matrix after the design_matrix transformation with the one from StatsModels
    @test size(mat) == size(mat2) == (100,12)
    @test size(mat3) == (100,4)
    
    # test the names of the columns of the design matrix
    @test mat_terms == ["(Intercept)",
                        "catvar1: 1",
                        "catvar2: B"
                        "catvar2: C",
                        "catvar2: D",
                        "catvar3: E",
                        "var4",
                        "var5",
                        "var6",
                        "var7"]
    @test mat_terms == mat2_terms
    @test mat3_terms == ["(Intercept)",
                         "catvar1: 1"
                         "catvar2: B"
                         "catvar2: C"
                         "catvar2: D"]
end 
