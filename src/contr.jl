"""
    get_dummy(df, cVar, cType. trtRef)

Convert categorical variable to dummy indicators using specified contrast 
type. This covers all cases except for treatment contrasts with a specified 
reference level. 

# Arguments 

- df = DataFrame of variables
- cVar = character string for the categorical variable in df to be converted
- cType = character string indicating the type of contrast to use for `cVar`
- trtRef = nothing

# Value

DataFrame of dummy variables for the specified categorical variable

"""
function get_dummy(df::DataFrames.DataFrame, cVar::String, cType::String, 
                   trtRef::Nothing)
    # Obtain the levels to use for the dummy indicators, depending on 
    # contrast type
    thisVar = string.(df[:,cVar])
    if cType=="treat"
        levs = unique(thisVar)[2:end]
    elseif (cType=="sum")
        levs = unique(thisVar)[(2:end).-1]
    elseif cType=="noint" || (cType=="sumnoint")
        levs = unique(thisVar)
    else
        error(string("Did not recognize contrast type for ", cVar))
    end
    
    # Iterate through levels to make dummy indicators
    namedict = Dict(zip(levs, 1:length(levs)))
    dummies = zeros(length(thisVar), length(namedict))
    for i=1:length(thisVar)
        if haskey(namedict, thisVar[i])
            dummies[i, namedict[thisVar[i]]] = 1
        end
    end
    
    # Some additional modifications for sum contrasts
    if (cType=="sum")
        dummies[thisVar.==(unique(thisVar)[end]),:] .= -1
    end
    if cType=="sumnoint"
        dummies = dummies - (1/length(unique(thisVar)))
    end
    
    # Convert results to a DataFrame and rename columns 
    newDf = convert(DataFrame, dummies)
    rename!(newDf, [Symbol("$(cVar)_$k") for k in levs])
    return newDf
end


"""

    get_dummy(df, cVar, cType. trtRef)

Convert categorical variables to for treatment contrasts with a specified 
reference level. 

# Arguments 

- df = DataFrame of variables
- cVar = character string for the categorical variable in df to be converted
- cType = character string indicating the type of contrast to use for `cVar`
- trtRef = character string specifying the level in cVar to use as the 
reference 

# Value

DataFrame of dummy variables for the specified categorical variable

"""
function get_dummy(df::DataFrames.DataFrame, cVar::String, cType::String, 
                   trtRef::String)
    
    # Obtain the levels to use for the dummy indicators.
    thisVar = string.(df[:,cVar])
    if cType=="treat"
        levs = unique(thisVar)[unique(thisVar) .!= trtRef]
    else
        error("Can only specify trtRef for treatment contrasts.")
    end
    
    # Iterate through levels to make dummy indicators
    namedict = Dict(zip(levs, 1:length(levs)))
    dummies = zeros(length(thisVar), length(namedict))
    for i=1:length(thisVar)
        if haskey(namedict, thisVar[i])
            dummies[i, namedict[thisVar[i]]] = 1
        end
    end
    
    # Convert results to a DataFrame and rename columns 
    newDf = convert(DataFrame, dummies)
    rename!(newDf, [Symbol("$(cVar)_$k") for k in levs])
    return newDf
end


"""
    contr(df, cVars, cTypes, trtRefs)

Converts categorical variables in a DataFrame to specified contrast types. 
All other variables are left as-is. 

# Arguments 

- df = DataFrame of variables
- cVars = 1d array of symbols corresponding to categorical variable names 
  in df to be converted
- cTypes = 1d array of character strings of the same length as `cVars`, 
  indicating the types of contrasts to use. Defaults to treatment contrasts 
  ("treat") for all variables in `cVars`. Other options include "sum" for sum 
  contrasts, "noint" for treatment contrasts with no intercept, and 
  "sumnoint" for sum contrasts with no intercept. For "treat" `cTypes`, you 
  can also specify the level to use as the reference treatment using `trtRefs`. 
- trtRefs = optional 1d array of character strings of the same length as 
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
function contr(df::DataFrames.DataFrame, cVars::AbstractArray{String,1}, 
               cTypes::AbstractArray{String,1}=repeat(["treat"], 
                                                      inner=length(cVars)), 
               trtRefs::AbstractArray{Union{Nothing, String},1}=
               repeat([nothing], inner=length(cVars)))  
   
    # If cVars only contains an empty Symbol, stop and return df
    if cVars == [Symbol()]
        return df
    end

    # Ensure that all Symbols in cVars are variables in df
    for cVar in cVars 
        if !in(cVar, names(df))
            error(string(cVar, " is not a variable in the DataFrame"))
        end
    end

    # Initialize new DataFrame
    newDf = DataFrame()
    
    # Iterate through variables in df
    for var in names(df)
        if !in(var, cVars)
            # Add non-categorical variables to the new DataFrame
            newDf[!,var] = df[:,var]
        else
            # Convert categorical variables to specified dummy contrasts
            dummyDf = get_dummy(df, var, cTypes[var.==cVars][1], 
                                trtRefs[var.==cVars][1])
            for dummy in names(dummyDf)
                newDf[!,dummy] = dummyDf[:,dummy]
            end
        end
    end
    
    return newDf
end
