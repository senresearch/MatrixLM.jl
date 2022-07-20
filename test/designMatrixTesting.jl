using DataFrames, Random, MatrixLM, StatsModels
n = 100
# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n),catvar3=rand(["D", "E"], n)), DataFrame(rand(n,4),:auto))

methods = Dict(:catvar1 => DummyCoding(), :catvar2 => EffectsCoding(base = "A"),:catvar3 =>DummyCoding())
mat = MatrixLM.design_matrix(@mlmFormula( catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4 ),X_df,
               [(:catvar1, :catvar3, DummyCoding()) , (:catvar2, EffectsCoding()) ]  )
mat2 = MatrixLM.design_matrix(@mlmFormula(catvar1 + catvar2 + catvar3 + x1 + x2 + x3 + x4),X_df, methods)