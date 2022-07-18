"""
    mlmFormula(ex)

    Capture and parse a formula expression for matrix linear model.

"""

macro mlmFormula(ex)
    name = string(ex)
    name = join(map(x -> isspace(name[x]) ? "" : name[x], 1:length(name)))
    return :(sum(term.(split($name, "+"))))
end

"""
    design_matrix(f, df::DataFrame,cntrst::Dict{Symbol, AbstractContrasts})

    Build design matrix.
    # Arguments 

    - f = formula for matrixLM, use @mlmFormula
    - df::DataFrames.DataFrame = DataFrame of variables
    - cntrst::Dict{Symbol, AbstractContrasts} = Encoding method for categorical or ordinal variables

"""

function design_matrix(;f, df::DataFrame,cntrst::Dict{Symbol, AbstractContrasts})
    return modelmatrix(f, df, hints= cntrst)
end