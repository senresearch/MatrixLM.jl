@testset "show methods" begin
    Y = randn(10, 5)
    X = hcat(ones(10), randn(10))
    Z = hcat(ones(5), randn(5))

    response = Response(Y)
    predictors = Predictors(X, Z, true, true)
    data = RawData(response, predictors)

    compact_data = sprint(show, data)
    detailed_data = sprint(
        show,
        MIME"text/plain"(),
        data,
    )

    @test compact_data == "RawData(n=10, m=5, p=2, q=2)"
    @test occursin("Response matrix Y: 10 × 5", detailed_data)
    @test occursin("X includes intercept: true", detailed_data)

    model = mlm(data)

    compact_model = sprint(show, model)
    detailed_model = sprint(
        show,
        MIME"text/plain"(),
        model,
    )

    @test compact_model == "Mlm(2 × 2 coefficient matrix)"
    @test occursin("Matrix linear model fit", detailed_model)
    @test occursin("Observations: 10", detailed_model)
    @test occursin("Coefficient matrix B: 2 × 2", detailed_model)
end