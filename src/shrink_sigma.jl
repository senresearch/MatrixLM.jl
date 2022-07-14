"""
    center(A::AbstractArray{Float64,2})

Centers columns of a 2d array

# Arguments

- A::AbstractArray{Float64,2} = 2d array of floats

# Value

2d array of floats

"""
function center(A::AbstractArray{Float64,2})
    
    # Row means of A
    m = mean(A, dims=1)
    # Initialize centered matrix
    W = zeros(size(A))
    
    # Subtract column means
    W .= A .- m
    
    return W
end


"""
    cov_est(resid::AbstractArray{Float64,2})

Estimates error variance and its variance/covariance

# Arguments

- resid::AbstractArray{Float64,2} = 2d array of floats consisting of the residuals

# Value

Tuple
- est: 2d array of floats; estimate
- varest: 2d array of floats; variance/covariance estimate

2d array of floats

"""
function cov_est(resid::AbstractArray{Float64,2})
    
    # Dimensions of residuals
    n = size(resid, 1)
    p = size(resid, 2)
    
    # Centered residual matrix
    W = center(resid)
    
    # Allocate space for the estimates and their variances
    est = zeros(p, p)
    varEst = zeros(p, p)
    
    # Loop through the possible entries
    for i = 1:p
        for j = i:p
            # Multiply the ith and jth columns
            ww = W[:,i].*W[:,j]
            
            if i==j # Diagonal elements
                est[i,i] = (n/(n-1)) * mean(ww)
                varEst[i,i] = (n/(n-1)^2) * var(ww)
            else # Non-diagonal elements
                est[i,j] = est[j,i] = (n/(n-1)) * mean(ww)
                varEst[i,j] = varEst[j,i] = (n/(n-1)^2) * var(ww)
            end
                
        end
    end
    
    return est, varEst
end


"""
    shrink_sigma(resid::AbstractArray{Float64,2}, targetType::String)

Estimates variance of errors and the shrinkage coefficient

# Arguments

- resid::AbstractArray{Float64,2} = 2d array of floats consisting of the residuals
- targetType::String = string indicating the target type toward which to shrink the 
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

# Reference

Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix 
    of stock returns with an application to portfolio selection. Journal of 
    empirical finance, 10(5), 603-621.

"""
function shrink_sigma(resid::AbstractArray{Float64,2}, targetType::String)
    
    # Dimensions of resid
    (n, p) = size(resid)
    
    # Estimates and the variance of the error variance
    (est, varEst) = cov_est(resid)
    
    if targetType=="A"  # Shrink to identity
        # Create identity target matrix
        T = Matrix{Float64}(I, p, p)
        # Estimate optimal lambda
        lambda = sum(varEst) / sum((est-T).^2)
        
    elseif targetType=="B"  # Shrink to common variance
        # Create target matrix
        T = Matrix{Float64}(I, p, p) * mean(diag(est))
        # Estimate optimal lambda
        lambda = sum(varEst) / sum((est-T).^2)
        
    elseif targetType=="C"  # Shrink to equal variance and covariance
        v = mean(diag(est))
        c = (sum(est) - sum(diag(est))) / (n*(n-1))
        # Create target matrix
        T = fill(c,(p,p)) + (v-c) * Matrix{Float64}(I, p, p)
        # Estimate optimal lambda
        lambda = sum(varEst) / sum((est-T).^2)
        
    elseif targetType=="D"  # Shrink to zero correlation
        v = diag(est)
        # Create target matrix
        T = diagm(0 => v)
        # Estimate optimal lambda
        lambda = (sum(varEst) - sum(diag(varEst))) /
                 (sum(est.^2) - sum(diag(est).^2))
    end
        
    return lambda*T + (1-lambda)*est, lambda
end
