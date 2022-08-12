"""
    Mlm(B::Array{Float64,2}, varB::Array{Float64,2}, sigma::Array{Float64,2},    
        data::RawData, weights, targetType, lambda::Float64)

Type for storing the results of an mlm model fit. 

"""
mutable struct Mlm
    
    # Coefficient estimates
    B::Array{Float64,2} 
    # Coefficient variance estimates
    varB::Array{Float64,2} 
    # Estimated variance of errors
    sigma::Array{Float64,2} 
    
    # Response and predictor matrices
    data::RawData 
    
    # Column weights for `Y`, or `nothing`
    weights 
    # String indicating target type to shrink the variance toward, or `nothing`
    targetType 
    # Estimated shrinkage coefficient
    lambda::Float64 
end


"""
    mlm_fit(data::RawData, weights::Nothing, targetType)

Matrix linear model using least squares method. Optionally incorporates 
shrinkage of the variance of the errors. 

# Arguments

- `data::RawData`: RawData object
- `weights::Nothing`: `nothing`
- `targetType`: string indicating the target type toward which to shrink the 
  error variance, or `nothing`. If the former, acceptable inputs are "A", "B", 
  "C", and "D". 
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries

# Value

An Mlm object

"""
function mlm_fit(data::RawData, weights::Nothing, targetType)
    
    # Calculate and store transpose(X)*X
    XTX = transpose(get_X(data))*get_X(data)
    # Calculate and store transpose(Z)*Z
    ZTZ = transpose(get_Z(data))*get_Z(data)
    
    # Estimate MLM coefficients
    B = calc_coeffs(get_X(data), get_Y(data), get_Z(data), XTX, ZTZ) 
    
    # Calculate residuals 
    resid = calc_resid(get_X(data), get_Y(data), get_Z(data), B)
    
    # Estimate variance of errors, optionally with variance shrinkage
    sigma, lambda = calc_sigma(resid, targetType)
    
    # Estimate variance of coefficient estimates
    varB = calc_var(get_X(data), get_Z(data), XTX, ZTZ, sigma)
    
    # Return Mlm object with estimates for coefficients, variances, and sigma
    return Mlm(B, varB, sigma, data, weights, targetType, lambda)
end


"""
    mlm_fit(data::RawData, weights::Array{Float64,1}, targetType)

Matrix linear model using column weighted least squares method. Optionally 
incorporates shrinkage of the variance of the errors. 

# Arguments

- `data`::RawData : RawData object
- `weights`::Array{Float64,1} : 1d array of floats to use as column weights for `Y`. Must be the 
  same length as the number of columns of `Y`. 
- `targetType` : string indicating the target type toward which to shrink the 
  error variance, or `nothing`. If the former, acceptable inputs are "A", "B", 
  "C", and "D". 
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries

# Value

An Mlm object

"""
function mlm_fit(data::RawData, weights::Array{Float64,1}, targetType)
    
    # Throw an error if the length of weights is incorrect
    if (length(weights) != size(get_Y(data),2)) 
        error("Weights must be same length as number of columns in Y.")
    end
    
    # Diagonal matrix with weights on diagonal.
    W = diagm(weights) 
    
    # Calculate and store WZ product
    WZ = W*get_Z(data)
    
    # Calculate and store transpose(X)*X
    XᵀX = transpose(get_X(data))*get_X(data)
    # Calculate and store transpose(Z)*W*Z
    ZᵀWᵀWZ = transpose(get_Z(data))*W*WZ
    
    # Estimate MLM coefficients
    B = calc_coeffs(get_X(data), get_Y(data), WZ, XᵀX, ZᵀWᵀWZ)
    
    # Calculate residuals 
    resid = calc_resid(get_X(data), get_Y(data), get_Z(data), B)
    
    # Estimate variance of errors, optionally with variance shrinkage
    sigma, lambda = calc_sigma(resid, targetType)
    
    # Estimate variance of coefficient estimates
    varB = calc_var(get_X(data), WZ, XᵀX, ZᵀWᵀWZ, sigma)
    
    # Return Mlm object with estimates for coefficients, variances, and sigma
    return Mlm(B, varB, sigma, data, weights, targetType, lambda)
end


"""
    mlm(data::RawData; addXIntercept::Bool=true, addZIntercept::Bool=true, weights=nothing, targetType=nothing)

Matrix linear model using least squares method. Column weighted least squares 
and shrinkage of the variance of the errors are options. 

# Arguments

- `data::RawData`: RawData object

# Keyword arguments

- `addXIntercept::Bool` : boolean flag indicating whether or not to include an `X` 
  intercept (row main effects). Defaults to `true`. 
- `addZIntercept::Bool` : boolean flag indicating whether or not to include a `Z` 
  intercept (column main effects). Defaults to `true`. 
- `weights` : 1d array of floats to use as column weights for `Y`, or `nothing`. 
  If the former, must be the same length as the number of columns of `Y`. 
  Defaults to `nothing`. 
- `targetType` : string indicating the target type toward which to shrink the 
  error variance, or `nothing`. If the former, acceptable inputs are "A", "B", 
  "C", and "D". Defaults to `nothing`. 
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries

# Value

An Mlm object

"""
function mlm(data::RawData; addXIntercept::Bool=true, addZIntercept::Bool=true, 
             weights=nothing, targetType=nothing, kwargs...)
    
    #= Deprecation:check for previous version keyword arguments
    ---------------------------------------------------------------------------------------------=#
    if haskey(kwargs, :isXIntercept)
        @warn "Keyword arguments `isXIntercept` and `isZIntercept` are deprecated, use 
                    `addXIntercept` and `addZIntercept` instead." 
              
        addXIntercept = values(kwargs).isXIntercept
    end

    if haskey(kwargs, :isZIntercept)
        @warn "Keyword arguments `isXIntercept` and `isZIntercept` are deprecated, use 
                    `addXIntercept` and `addZIntercept` instead."
                    
        addZIntercept = values(kwargs).isZIntercept
    end
    #-----------------------------------------------------------------------------------------------

    # Add X and Z intercepts if necessary
    if addXIntercept==true && data.predictors.hasXIntercept==false
        data.predictors.X = add_intercept(data.predictors.X)
        data.predictors.hasXIntercept = true
        data.p = data.p + 1
    end
    if addZIntercept==true && data.predictors.hasZIntercept==false
        data.predictors.Z = add_intercept(data.predictors.Z)
        data.predictors.hasZIntercept = true
        data.q = data.q + 1
    end
    
    # Remove X and Z intercepts in new predictors if necessary
    if addXIntercept==false && data.predictors.hasXIntercept==true
        data.predictors.X = remove_intercept(data.predictors.X)
        data.predictors.hasXIntercept = false
        data.p = data.p - 1
    end
    if addZIntercept==false && data.predictors.hasZIntercept==true
        data.predictors.Z = remove_intercept(data.predictors.Z)
        data.predictors.hasZIntercept = false
        data.q = data.q - 1
    end

    if (typeof(targetType) != Nothing) & !(targetType in ["A", "B", "C", "D"])
        println("Unrecognizable targetType will be ignored and no variance shrinkage will be performed.")
        targetType = nothing
    end
    
    # Run matrix linear models
    return mlm_fit(data, weights, targetType)
end


"""
    t_stat(MLM::Mlm, isMainEff::Bool=false)

Calculates t-statistics of an Mlm object

# Arguments 

- `MLM::Mlm `: Mlm object
- `isMainEff::Bool` : boolean flag indicating whether or not to include t-statistics 
  for the main effects

# Value

2d array of floats

"""
function t_stat(MLM::Mlm, isMainEff::Bool=false)
  
    # Cases when not including main effects
    if isMainEff== false 
        # If X and Z intercepts were both included
        if (MLM.data.predictors.hasXIntercept==true) && 
           (MLM.data.predictors.hasZIntercept==true) 
            return MLM.B[2:end, 2:end]./sqrt.(MLM.varB[2:end, 2:end])
            
        # If only X intercept was included
        elseif MLM.data.predictors.hasXIntercept==true 
            return MLM.B[2:end, :]./sqrt.(MLM.varB[2:end, :])
            
        # If only Z intercept was included
        elseif  MLM.data.predictors.hasZIntercept==true 
            return MLM.B[:, 2:end]./sqrt.(MLM.varB[:, 2:end])
            
        end
    end
	
    # Case when including main effects
    return MLM.B ./ sqrt.(MLM.varB)
end
