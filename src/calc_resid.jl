"""
    calc_resid(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
           Z::AbstractArray{Float64,2}, B::AbstractArray{Float64,2})

Calculate residuals

# Arguments 

- X = 2d array of floats consisting of the row covariates, standardized as 
  necessary
- Y = 2d array of floats consisting of the multivariate response 
  observations, standardized as necessary
- Z = 2d array of floats consisting of the column covariates, standardized 
  as necessary
- B = 2d array of floats consisting of coefficient estimates

# Value 

2d array of floats

"""
function calc_resid(X::AbstractArray{Float64,2}, Y::AbstractArray{Float64,2}, 
                    Z::AbstractArray{Float64,2}, B::AbstractArray{Float64,2})
    
    # Obtain fitted values
    resid = calc_preds(X, Z, B) 
    
    # Compute residuals over fitted values
    resid .= Y .- resid
    
    return resid
end


"""
    calc_resid!(resid::AbstractArray{Float64,2}, 
                     X::AbstractArray{Float64,2}, 
                     Y::AbstractArray{Float64,2}, 
                     Z::AbstractArray{Float64,2}, 
                     B::AbstractArray{Float64,2})

Calculate residuals in place

# Arguments 

- resid = 2d array of floats consisting of the residuals, to be updated in 
  place
- X = 2d array of floats consisting of the row covariates, standardized as 
  necessary
- Y = 2d array of floats consisting of the multivariate response 
  observations, standardized as necessary
- Z = 2d array of floats consisting of the column covariates, standardized 
  as necessary
- B = 2d array of floats consisting of coefficient estimates

# Value 

None; updates residuals in place. 

"""
function calc_resid!(resid::AbstractArray{Float64,2}, 
                     X::AbstractArray{Float64,2}, 
                     Y::AbstractArray{Float64,2}, 
                     Z::AbstractArray{Float64,2}, 
                     B::AbstractArray{Float64,2})
    
    # Obtain fitted values in place
    calc_preds!(resid, X, Z, B) 
    
    # Compute residuals in place over fitted values
    resid .= Y .- resid
end