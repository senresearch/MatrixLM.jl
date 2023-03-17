using Random, MatrixLM, DataFrames, Test, RecipesBase

Random.seed!(1)

# Dimensions of matrices 
n = 100
m = 250

# Number of column covariates
q = 20

# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), DataFrame(rand(n,4),:auto))
# Convert dataframe to predicton matrix
X = Matrix(contr(X_df, [:catvar1, :catvar2], ["treat", "sum"]))

p = size(X)[2]
Z = rand(m,q)
B = rand(-5:5,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E
# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z));
est = mlm(dat);

tStats = t_stat(est);

rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), MLMplots(tStats, 2 ,["a" "aa" "d" "s" "sv" "zx" "eq" "j" "m" "o" ]))

# Plot the t-statistics of the coefficients
@testset "recipe plot test" begin
    @test rec[1].args[1] == tStats[:,2]
    @test rec[1].plotattributes[:xticks][2] == ["a" "aa" "d" "s" "sv" "zx" "eq" "j" "m" "o"]
end

# Notes: Alternative testing could compare plot images by using Image.jl.
