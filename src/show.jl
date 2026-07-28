"""
    show(io::IO, data::RawData)

Display a compact summary of a `RawData` object.
"""
function Base.show(io::IO, data::RawData)
    print(
        io,
        "RawData(n=$(data.n), m=$(data.m), p=$(data.p), q=$(data.q))",
    )
end

"""
    show(io::IO, ::MIME"text/plain", data::RawData)

Display a readable summary of the matrices and dimensions stored in a
`RawData` object.
"""
function Base.show(io::IO, ::MIME"text/plain", data::RawData)
    println(io, "RawData")
    println(io, "  Response matrix Y: $(data.n) × $(data.m)")
    println(io, "  Design matrix X: $(data.n) × $(data.p)")
    println(io, "  Design matrix Z: $(data.m) × $(data.q)")
    println(io, "  X includes intercept: $(data.predictors.hasXIntercept)")
    print(io, "  Z includes intercept: $(data.predictors.hasZIntercept)")
end

"""
    show(io::IO, model::Mlm)

Display a compact summary of a fitted matrix linear model.
"""
function Base.show(io::IO, model::Mlm)
    print(
        io,
        "Mlm($(size(model.B, 1)) × $(size(model.B, 2)) coefficient matrix)",
    )
end

"""
    show(io::IO, ::MIME"text/plain", model::Mlm)

Display a readable summary of a fitted matrix linear model.
"""
function Base.show(io::IO, ::MIME"text/plain", model::Mlm)
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
        "  Residual covariance matrix: " *
        "$(size(model.sigma, 1)) × $(size(model.sigma, 2))",
    )
    println(io, "  Weighted fit: $(model.weights !== nothing)")

    if model.targetType === nothing
        print(io, "  Covariance shrinkage: none")
    else
        println(io, "  Covariance shrinkage target: $(model.targetType)")
        print(io, "  Shrinkage coefficient: $(model.lambda)")
    end
end