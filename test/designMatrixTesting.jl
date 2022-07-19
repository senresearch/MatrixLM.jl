n = 100
m = 250
# Number of column covariates
q = 20
# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n),catvar3=rand(["D", "E"], n)), DataFrame(rand(n,4),:auto))

#methods = Dict(:catvar1 => DummyCoding(), :catvar2 => EffectsCoding(base = "A"))
design_matrix2(f=@mlmFormula(catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4),df=X_df,
               cntrstArray=[(:catvar1, DummyCoding()) (:catvar2, :catvar3, EffectsCoding()) ]  )