"""
    mlmFormula(ex)

    Capture and parse a formula expression for matrix linear model.

"""
macro mlmFormula(ex)
    name = join(map(x -> isspace(string(ex)[x]) ? "" : string(ex)[x], 1:length(string(ex))))
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

"""

    design_matrix(;f, df::DataFrame,cntrstArray::Array)

    Build design matrix.
    # Arguments 

    - f = formula for matrixLM, use @mlmFormula
    - df::DataFrames.DataFrame = DataFrame of variables
    - cntrstArray = An array containing tuples of variable and its encoding function.

"""
function design_matrix(;f, df::DataFrame, cntrst::Matrix)
    cntrsts = Dict{Symbol, AbstractContrasts}()
    for cntrsTuple in cntrst
        for i in 1:length(cntrsTuple)-1
            fun = cntrsTuple[length(cntrsTuple)]
            cntrsts[cntrsTuple[i]] = fun
        end
    end    
    return modelmatrix(f, df, hints= cntrsts)
end