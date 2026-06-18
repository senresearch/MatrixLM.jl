"""
    center(A::AbstractArray{Float64,2})

Centers columns of a 2d array

# Arguments

- `A::AbstractArray{Float64,2}`: 2d array of floats

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

- `resid::AbstractArray{Float64,2}`: 2d array of floats consisting of the residuals

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

- `resid::AbstractArray{Float64,2}`: 2d array of floats consisting of the residuals
- `targetType::String`: string indicating the target type toward which to shrink the 
  variance. Acceptable inputs are "A", "B", "C", and "D". 
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries

# Value

Tuple
- `sigma`: 2d array of floats; shrunk estimated variance of errors
- `lambda`: floating scalar; estimated shrinkage coefficient 
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

# ----------------------------------------------------------------------

"""
    tri2vec(X::Matrix{Float64}

Extracts the diagonal and the upper triangular elements of a square
matrix into two vectors.

# Value

A tuple of two vectors with the off diagonal and diagonal
respectively.

# See also

`tri2vec`
"""
function tri2vec(X::Matrix{Float64})
    m = size(X,1)
    if(size(X,2)!=m)
        error("Matrix not square.")
    end
    x = zeros(div(m*(m-1),2))
    d = zeros(m)
    
    idx = 1
    for i in 1:(m-1)
        for j in ((i+1):m)
            x[idx] = X[i,j]
            idx+=1
        end
    end

    for i in 1:m
        d[i] = X[i,i]
    end
    return (x,d)
end

"""
    vec2tri(X::Matrix{Float64}

Reconstructs a square matrix from two vectors which are the upper
triangular elements and the diagonal respectively.

# Value

Square symmetric matrix.

# See also

`vec2tri`
"""
function vec2tri(x::Vector{Float64},d::Vector{Float64})
    m = (isqrt(8*length(x)+1)+1) ÷ 2
    X = zeros(m,m)

    if( m!=length(d) )
        error("Incompatible dimensions of the two vectors.")
    end
    
    idx = 1
    for i in 1:(m-1)
        for j in (i+1):m
            X[i,j] = X[j,i] =  x[idx]
            idx+=1
        end
    end

    for i in 1:m
        X[i,i] = d[i]
    end
    
    return X
end

"""
    shrink(x::Vector{Float64},t::Float64=0.0,v::Float64=1.0)

Calculates shrinkage estimates based on estimates and their variances
towards a target.

# Arguments

- `x::Vector{Float64}`: Vector of estimates
- `t::Float64=0`: Target to shrink to
- `v::Float64=1.0`: Common variance of the estimates

# Value

Tuple:
- `x::Vector{Float64}`: Shrunk estimates
- `λ:Float64`: Shrinkage coefficient (0=no shrinkage; 1=complete shrinkage) 

"""
function shrink(x::Vector{Float64},t::Float64=0.0,v::Float64=1.0)
    k = length(x)
    λ = k*v / sum((x .- t).^2)
    λ = minimum((1.0,λ))
    return λ*t .+ (1-λ) .* x, λ
end

"""
    shrink_var(resid::Matrix{Float64})

Estimates covariance matrix of residuals using a shrinkage estimator

# Arguments

- `resid::Matrix{Flaot64}`: 2d array of floats consisting of the residuals

# Value

Tuple
- `sigma`: 2d array of floats; shrunk estimated variance of errors
- `lambda`: floating scalar; estimated shrinkage coefficient 
  (0 = no shrinkage, 1 = complete shrinkage)

# Details

The covariance matrix is decomposed into a correlation matrix and a variance
vector.  They are each transformed by a variance stabilizing transform and
shrunk towards a common value. Then they are transformed back and the
covariance matrix is reconstructed.

# Reference

Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix 
    of stock returns with an application to portfolio selection. Journal of 
    empirical finance, 10(5), 603-621.

"""
function shrink_var(X::Matrix{Float64})

    (n,m) = size(X) # number of samples
    if (n<=3) 
        error("Sample size too small for proceeding further.")
    end

    # variance of correlation and variance after variance stabilizing transform
    vr = 1/(n-3)
    vv = 2/(n-2)

    
    (r,a) = tri2vec(cor(X))
    r = atanh.(r)
    v = log.(vec(var(X,dims=1)))

    (λr,r) = shrink(r,mean(r),vr)
    (λv,v) = shrink(v,mean(v),vv)

    r[:] = tanh.(r)
    v[:] = exp.(v.+0.5.*(var(v).+mean(vv)))
    s = sqrt.(v)

    R = vec2tri(r,a)
    
    S = Diagonal(s) * R * Diagonal(s)
    
end

# checks to perform

# - tri2vec and vec2tri should be inverses of each other
# - tri2vec should only work for square matrices
# - vec2tri should have compatible argument lengths
# - shrunk covariance should be positive definite even if input is not

