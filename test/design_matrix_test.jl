
###########################
# Generate Simulated Data #
###########################
n = 100
# Generate data with two categorical variables and 4 numerical variables.
X_df = hcat(
	DataFrame(
		catvar1 = string.(rand(0:1, n)),
		catvar2 = rand(["A", "B", "C", "D"], n),
		catvar3 = rand(["D", "E"], n),
	),
	DataFrame(rand(n, 4), ["var1", "var2", "var3", "var4"]),
);


my_contrasts = Dict(
	:catvar1 => DummyCoding(),
	:catvar2 => EffectsCoding(base = "A"),
	:catvar3 => DummyCoding(base = "E"),
)

# design matrix including categorical and continuous variables           
mat_1 = MatrixLM.design_matrix(
	@mlmformula(1 + catvar1 + catvar2 + catvar3 + var1 + var2 + var3 + var4),
	X_df,
	my_contrasts,
)

my_contrasts_vec = [
	(:catvar1, DummyCoding()),
	(:catvar2, EffectsCoding(base = "A")),
	(:catvar3, DummyCoding(base = "E")),
]

mat_1_vec = MatrixLM.design_matrix(
	@mlmformula(1 + catvar1 + catvar2 + catvar3 + var1 + var2 + var3 + var4),
	X_df,
	my_contrasts_vec,
)

# design matrix including only categorical without spcifying my_contrasts
# default is dummy coded
mat_2 = MatrixLM.design_matrix(
	@mlmformula(1 + catvar1 + catvar2),
	X_df,
)

# get the columns names of the design matrix
mat_1_terms = MatrixLM.design_matrix_names(
	@mlmformula(1 + catvar1 + catvar2 + catvar3 + var1 + var2 + var3 + var4),
	X_df,
	[(:catvar1, :catvar3, DummyCoding()), (:catvar2, EffectsCoding(base = "B"))],
)

# get the columns names of the design matrix
mat_1_terms_b = MatrixLM.design_matrix_names(
	@mlmformula(1 + catvar1 + catvar2 + catvar3 + var1 + var2 + var3 + var4),
	X_df,
	my_contrasts,
)

# get the columns names of the design matrix
mat_2_terms = MatrixLM.design_matrix_names(
	@mlmformula(1 + catvar1 + catvar2),
	X_df,
)
########
# Test #
########

@testset "designMatrixTesting" begin
	# test the dimension of the matrix after the design_matrix transformation with the one from StatsModels
	@test size(mat_1) == (100, 10)
	@test size(mat_2) == (100, 5)

	# test the vector-based contrast specification matches the dictionary version
	@test mat_1_vec == mat_1

	# test the names of the columns of the design matrix
	@test mat_1_terms == ["(Intercept)",
		"catvar1: 1",
		"catvar2: A",
		"catvar2: C",
		"catvar2: D",
		"catvar3: E",
		"var1",
		"var2",
		"var3",
		"var4"]
	@test mat_1_terms_b == ["(Intercept)",
		"catvar1: 1",
		"catvar2: B",
		"catvar2: C",
		"catvar2: D",
		"catvar3: D",
		"var1",
		"var2",
		"var3",
		"var4"]
	@test mat_2_terms == ["(Intercept)",
		"catvar1: 1",
		"catvar2: B",
		"catvar2: C",
		"catvar2: D"]
end
