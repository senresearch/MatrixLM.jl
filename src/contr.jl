"""
    get_dummy(df::DataFrames.DataFrame, cVar::Symbol, 
              cType::String, trtRef::Union{Nothing,String}=nothing)

Convert a categorical variable to dummy indicators using the specified
contrast type.

# Arguments

- `df::DataFrames.DataFrame`: DataFrame containing the variables.
- `cVar::Symbol`: column name in `df` for the categorical variable.
- `cType::String`: contrast type to use. Supported values are
  "treat", "sum", "noint", and "sumnoint".
- `trtRef::Union{Nothing,String}`: optional reference level for "treat"
  or "sum" contrasts. When omitted (`nothing` or an empty string),
  the reference defaults to the last sorted level for "treat" contrasts
  and the first sorted level for "sum" contrasts.

# Value

DataFrame containing the dummy variables for the specified categorical
variable.

"""
function get_dummy(df::DataFrames.DataFrame, cVar::Symbol, cType::String,
                   trtRef::Union{Nothing,String}=nothing)

    # Convert levels to strings and compute sorted unique values once
    thisVar = string.(df[:, cVar])
    sortedLevs = sort(unique(thisVar))

    isempty(sortedLevs) && return DataFrame()

    # Normalise the optional reference argument
    refProvided = !(trtRef === nothing || (isa(trtRef, String) && isempty(trtRef)))
    reference = nothing

    if cType == "treat" || cType == "sum"
        if refProvided
            reference = trtRef::String
            if !(reference in sortedLevs)
                error("Reference level '" * reference * "' not found in variable " * string(cVar) *
                      ". Available levels: " * join(sortedLevs, ", "))
            end
        else
            reference = cType == "treat" ? sortedLevs[1] : sortedLevs[end]
        end
    elseif refProvided
        error("A reference level can only be provided for \"treat\" or \"sum\" contrasts.")
    end

    # Determine levels used to create indicator columns
    levs = if cType == "treat"
        filter(x -> x != reference, sortedLevs)
    elseif cType == "sum"
        filter(x -> x != reference, sortedLevs)
    elseif cType == "noint" || cType == "sumnoint"
        sortedLevs
    else
        error("Did not recognize contrast type '" * cType * "' for " * string(cVar) *
              ". Valid options: \"treat\", \"sum\", \"noint\", \"sumnoint\"")
    end

    namedict = Dict(zip(levs, 1:length(levs)))
    dummies = zeros(Float64, length(thisVar), length(namedict))

    for i in 1:length(thisVar)
        if haskey(namedict, thisVar[i])
            dummies[i, namedict[thisVar[i]]] = 1.0
        end
    end

    if cType == "sum" && reference !== nothing
        refMask = thisVar .== reference
        if any(refMask) && !isempty(levs)
            dummies[refMask, :] .= -1.0
        end
    elseif cType == "sumnoint"
        dummies .= dummies .- (1.0 / length(sortedLevs))
    end

    newDf = DataFrame(dummies, :auto)
    rename!(newDf, [Symbol("$(cVar)_$k") for k in levs])

    return newDf
end


"""
    contr(df::DataFrames.DataFrame, cVars::AbstractArray{Symbol,1}, 
          cTypes::AbstractArray{String,1}=repeat(["treat"], inner=length(cVars)), 
          trtRefs::AbstractArray= repeat([nothing], inner=length(cVars))) 

Converts categorical variables in a DataFrame to specified contrast types. 
All other variables are left as-is. 

# Arguments 

- `df::DataFrames.DataFrame`: DataFrame of variables
- `cVar::Symbol`: symbol for the categorical variable in df to be converted
- `cTypes::AbstractArray{String,1}`: 1d array of character strings of the same length as `cVars`, 
  indicating the types of contrasts to use. Defaults to treatment contrasts 
  ("treat") for all variables in `cVars`. Other options include "sum" for sum 
  contrasts, "noint" for treatment contrasts with no intercept, and 
  "sumnoint" for sum contrasts with no intercept. For "treat" `cTypes`, you 
  can also specify the level to use as the reference treatment using `trtRefs`. 
- `trtRefs::AbstractArray`: optional 1d array of character strings of the same length as 
  `cVars`, specifying the level to use as the references for treatment 
  contrasts. Defaults to nothing for all variables in `cVars`.
	
# Value

DataFrame with same variables as the original DataFrame, but categorical 
variables converted to dummy contrasts. 

# Some notes

If `cVars` consists of only an empty Symbol, i.e. `cVars=[Symbol()]`, this 
will signal to the function that no contrasts should be created. The 
original DataFrame will be returned. 

"""
function contr(df::DataFrames.DataFrame, cVars::AbstractArray{Symbol,1}, 
               cTypes::AbstractArray{String,1}=repeat(["treat"], 
                                                      inner=length(cVars)), 
               trtRefs::AbstractArray= repeat([nothing], inner=length(cVars))) 
   
    # If cVars only contains an empty Symbol, stop and return df
    if cVars == [Symbol()]
        return df
    end

    # Ensure that all Symbols in cVars are variables in df
    for cVar in cVars 
        if !in(cVar, propertynames(df))
            error(string(cVar, " is not a variable in the DataFrame"))
        end
    end

    # Initialize new DataFrame
    newDf = DataFrame()
    
    # Iterate through variables in df
    for var in propertynames(df)
        if !in(var, cVars)
            # Add non-categorical variables to the new DataFrame
            newDf[!,var] = df[:,var]
        else
            # Convert categorical variables to specified dummy contrasts
            dummyDf = get_dummy(df, var, cTypes[var.==cVars][1], 
                                trtRefs[var.==cVars][1])
            for dummy in propertynames(dummyDf)
                newDf[!,dummy] = dummyDf[:,dummy]
            end
        end
    end
    
    return newDf
end

