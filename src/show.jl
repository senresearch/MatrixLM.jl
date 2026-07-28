function row_preview(mat::AbstractMatrix; max_cols::Int=3)
    if size(mat, 1) == 0 || size(mat, 2) == 0
        return "[]"
    end

    ncols = min(size(mat, 2), max_cols)
    values = [string(mat[1, j]) for j in 1:ncols]
    preview = "[" * join(values, ", ") * (size(mat, 2) > ncols ? ", …" : "") * "]"
    return preview
end

"""
    show(io::IO, data::RawData)

Display a readable summary of the matrices and dimensions stored in a
`RawData` object.
"""
function Base.show(io::IO, data::RawData)
    println(io, "RawData")
    println(io, "  Response matrix Y: $(data.n) × $(data.m)")
    println(io, "  Design matrix X: $(data.n) × $(data.p)")
    println(io, "  Design matrix Z: $(data.m) × $(data.q)")
    println(io, "  X includes intercept: $(data.predictors.hasXIntercept)")
    println(io, "  Z includes intercept: $(data.predictors.hasZIntercept)")
    println(io, "  Preview of Y first row (first $(min(3, size(get_Y(data), 2))) columns): " *
        row_preview(round.(get_Y(data), digits=4)))
    println(io, "  Preview of X first row (first $(min(3, size(get_X(data), 2))) columns): " *
        row_preview(round.(get_X(data), digits=4)))
    println(io, "  Preview of Z first row (first $(min(3, size(get_Z(data), 2))) columns): " *
        row_preview(round.(get_Z(data), digits=4)))
end

# """
#     show(io::IO, model::Mlm)

# Display a compact summary of a fitted matrix linear model.
# """
# function Base.show(io::IO, model::Mlm)
#     print(
#         io,
#         "Mlm($(size(model.B, 1)) × $(size(model.B, 2)) coefficient matrix)",
#     )
# end

"""
    show(io::IO, model::Mlm)

Display a readable summary of a fitted matrix linear model.
"""
function Base.show(io::IO, model::Mlm)
    data = model.data

    println(io, "Matrix linear model fit")
    println(io, "  Observations: $(data.n)")
    println(io, "  Responses: $(data.m)")
    println(io, "  Row predictors: $(data.p)")
    println(io, "  Column predictors: $(data.q)")
    println(
        io,
        "  Coefficient matrix B: " *
        "$(size(model.B, 1)) × $(size(model.B, 2))",
    )
    println(
        io,
        "  Residual covariance matrix sigma(Σ): " *
        "$(size(model.sigma, 1)) × $(size(model.sigma, 2))",
    )
    println(io, "  Weighted fit: $(model.weights !== nothing)")
    println(
        io,
        "  Preview of B first row (first $(min(3, size(model.B, 2))) columns): " *
        row_preview(round.(model.B, digits=4)),
    )
    println(
        io,
        "  Preview of sigma(Σ) first row (first $(min(3, size(model.sigma, 2))) columns): " *
        row_preview(round.(model.sigma, digits=4)),
    )

    if model.targetType === nothing
        println(io, "  Covariance shrinkage: none")
    else
        println(io, "  Covariance shrinkage target: $(model.targetType)")
        println(io, "  Shrinkage coefficient: (correlation = $(round(model.lambda.correlation,
         digits=4)), variance = $(round(model.lambda.variance, digits=4)))")
    end
end