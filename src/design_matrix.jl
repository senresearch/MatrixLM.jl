"""
    mlmFormula(ex)

    Capture and parse a formula expression for matrix linear model.

"""
macro mlmFormula(ex)
    ex_string = "0 ~" * string(ex) 
    return @eval(@formula($(Meta.parse(ex_string))).rhs)
end

"""
    design_matrix(f, df::DataFrame,cntrst::Dict{Symbol, AbstractContrasts})

    Build design matrix.
    # Arguments 

    - f = formula for matrixLM, use @mlmFormula
    - df::DataFrames.DataFrame = DataFrame of variables
    - cntrst::Dict{Symbol, AbstractContrasts} = Encoding method for categorical or ordinal variables

"""
function design_matrix(f, df::DataFrame,cntrst::Dict{Symbol, AbstractContrasts})
    return modelmatrix(f, df, hints= cntrst)
end

"""

    design_matrix(f, df::DataFrame, cntrst::Vector)

    Build design matrix.
    # Arguments 

    - f = formula for matrixLM, use @mlmFormula
    - df::DataFrames.DataFrame = DataFrame of variables
    - cntrst = An vactor containing tuples of variable and its encoding function.

"""
function design_matrix(f, df::DataFrame, cntrst::Vector)
    cntrsts = Dict{Symbol, AbstractContrasts}()
    for cntrsTuple in cntrst
        fun = cntrsTuple[length(cntrsTuple)]
        for i in 1:length(cntrsTuple)-1
            cntrsts[cntrsTuple[i]] = fun
        end
    end    
    return modelmatrix(f, df, hints= cntrsts)
end