
@testset "row_preview" begin
    empty_mat = zeros(0, 0)
    single_row_mat = [1.0 2.0 3.0]
    wide_mat = reshape(1.0:8.0, (2, 4))

    @test MatrixLM.row_preview(empty_mat) == "[]"
    @test MatrixLM.row_preview(single_row_mat) == "[1.0, 2.0, 3.0]"
    @test MatrixLM.row_preview(wide_mat) == "[1.0, 2.0, 3.0, …]"
    @test MatrixLM.row_preview(wide_mat; max_cols=2) == "[1.0, 2.0, …]"
end

@testset "show methods" begin
    Y = randn(10, 5)
    X = hcat(ones(10), randn(10))
    Z = hcat(ones(5), randn(5))

    response = Response(Y)
    predictors = Predictors(X, Z, true, true)
    data = RawData(response, predictors)

    detailed_data = sprint(show, MIME"text/plain"(), data)

    @test occursin("RawData", detailed_data)
    @test occursin("Response matrix Y: 10 × 5", detailed_data)
    @test occursin("Design matrix X: 10 × 2", detailed_data)
    @test occursin("Design matrix Z: 5 × 2", detailed_data)
    @test occursin("X includes intercept: true", detailed_data)
    @test occursin("Z includes intercept: true", detailed_data)
    @test occursin("Preview of Y first row", detailed_data)
    @test occursin("Preview of X first row", detailed_data)
    @test occursin("Preview of Z first row", detailed_data)

    model = mlm(data)

    detailed_model = sprint(show, MIME"text/plain"(), model)

    @test occursin("Matrix linear model fit", detailed_model)
    @test occursin("Observations: 10", detailed_model)
    @test occursin("Responses: 5", detailed_model)
    @test occursin("Row predictors: 2", detailed_model)
    @test occursin("Column predictors: 2", detailed_model)
    @test occursin("Coefficient matrix B: 2 × 2", detailed_model)
    @test occursin("Residual covariance matrix sigma(Σ): 5 × 5", detailed_model)
    @test occursin("Weighted fit: false", detailed_model)
    @test occursin("Preview of B first row", detailed_model)
    @test occursin("Preview of sigma(Σ) first row", detailed_model)
    @test occursin("Covariance shrinkage: none", detailed_model)
end