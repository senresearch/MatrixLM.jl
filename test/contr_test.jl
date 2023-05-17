###########
# Library #
###########
using DataFrames, Random, MatrixLM, StatsModels

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
