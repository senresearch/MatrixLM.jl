"""

    calc_coeffs(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, Z::AbstractArray{Float64,2}, XTX::AbstractArray{Float64,2}, ZTZ::AbstractArray{Float64,2})

Calculates the the coefficient estimates

# Arguments

- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Y = 2d array of floats consisting of the multivariate response
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- XTX = X*transpose(X) product as a 2d array of floats 
- ZTZ = Z*transpose(Z) product as a 2d array of floats 

# Value

2d array of floats

"""
function calc_coeffs(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
                     Z::AbstractArray{Float64,2}, 
                     XTX::AbstractArray{Float64,2}, 
                     ZTZ::AbstractArray{Float64,2})
    
    return transpose(ZTZ\(transpose((XTX\transpose(X)*Y)*Z)))
end


"""
    calc_sigma(resid::AbstractArray{Float64,2}, targetType::Nothing)

Estimates variance of errors and the shrinkage coefficient, without variance 
shrinkage. 

# Arguments

- resid = 2d array of floats consisting of the residuals
- targetType =`nothing`

# Value

Tuple
- sigma: 2d array of floats; estimated variance of errors
- lambda: 0.0

# Some notes

Since this version of `calc_sigma` does not implement variance shrinkage, the 
shrinkage coefficient lambda is 0. 

"""
function calc_sigma(resid::AbstractArray{Float64,2}, targetType::Nothing)

    # Residual sum of squares
    RSS = transpose(resid)*resid
    # Divide by the number of samples
    sigma = RSS ./ (size(resid,1) - 1)
    
    return sigma, 0.0
end


"""
    calc_sigma(resid::AbstractArray{Float64,2}, targetType::AbstractString)

Estimates variance of errors and the shrinkage coefficient, with variance 
shrinkage. 

# Arguments

- resid = 2d array of floats consisting of the residuals
- targetType = string indicating the target type toward which to shrink the 
  variance. Acceptable inputs are "A", "B", "C", and "D". 
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries

# Value

Tuple
- sigma: 2d array of floats; shrunk estimated variance of errors
- lambda: floating scalar; estimated shrinkage coefficient 
  (0 = no shrinkage, 1 = complete shrinkage)

"""
function calc_sigma(resid::AbstractArray{Float64,2}, 
                    targetType::AbstractString) 
    
    return shrink_sigma(resid, targetType) 
end


"""
    calc_var(X::AbstractArray{Float64,2}, Z::AbstractArray{Float64,2}, XTX::AbstractArray{Float64,2}, ZTZ::AbstractArray{Float64,2}, sigma::AbstractArray{Float64,2})

Calculate the variance (diagonal of the covariance matrix) of the coefficient 
estimates. 

# Arguments

- X = 2d array of floats consisting of the row covariates, with all 
  categorical variables coded in appropriate contrasts
- Z = 2d array of floats consisting of the column covariates, with all 
  categorical variables coded in appropriate contrasts
- XTX = X*transpose(X) product as a 2d array of floats 
- ZTZ = Z*transpose(Z) product as a 2d array of floats 
- sigma = 2d array of floats consisting of the estimated sigma

# Value

2d array of floats

"""
function calc_var(X::AbstractArray{Float64,2}, Z::AbstractArray{Float64,2}, 
                  XTX::AbstractArray{Float64,2}, 
                  ZTZ::AbstractArray{Float64,2}, 
                  sigma::AbstractArray{Float64,2})
    
    # LHS of covariance matrix 
    varLeft = inv(XTX) 
    # RHS of covariance matrix
    varRight = transpose(ZTZ\(transpose((ZTZ\(transpose(Z)*sigma))*Z))) 
    
    # Diagonal of covariance matrix, aka the variance
    varDiag = transpose(kron_diag(varLeft, varRight)) 

    return varDiag
end