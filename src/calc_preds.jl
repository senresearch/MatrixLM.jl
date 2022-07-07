"""
    calc_preds(X::AbstractArray{Float64,2}, 
               Z::AbstractArray{Float64,2}, 
               B::AbstractArray{Float64,2})

Predict values

# Arguments 

- X::AbstractArray{Float64,2} = The row covariates, standardized as 
  necessary
- Z::AbstractArray{Float64,2} = The column covariates, standardized 
  as necessary
- B::AbstractArray{Float64,2} = Coefficient estimates

# Value 

2d array of floats

"""
function calc_preds(X::AbstractArray{Float64,2}, 
                    Z::AbstractArray{Float64,2}, 
                    B::AbstractArray{Float64,2})
    
    # Predict new values
    X*B*transpose(Z)
end


"""
    calc_preds!(preds::AbstractArray{Float64,2}, 
                X::AbstractArray{Float64,2}, 
                Z::AbstractArray{Float64,2}, 
                B::AbstractArray{Float64,2})

Predict values in place

# Arguments 

- preds::AbstractArray{Float64,2}: The predicted values, to be 
  updated in place
- X::AbstractArray{Float64,2}: The row covariates, standardized as 
  necessary
- Z::AbstractArray{Float64,2}: The column covariates, standardized 
  as necessary
- B::AbstractArray{Float64,2}: Coefficient estimates

# Value 

None; updates predicted values in place. 

"""
function calc_preds!(preds::AbstractArray{Float64,2}, 
                     X::AbstractArray{Float64,2}, 
                     Z::AbstractArray{Float64,2}, 
                     B::AbstractArray{Float64,2})
    
    # Update predicted values in place
    mul!(preds, X*B, transpose(Z)) 
end