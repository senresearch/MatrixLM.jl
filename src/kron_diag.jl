"""
    diagonal(A)

Get the diagonal of a 2d array of floats. This just calls the base diag 
function.

# Arguments

- A = 2d array of floats

# Value 

1d array of floats

"""
function diagonal(A::AbstractArray{Float64,2})
    
    return LinearAlgebra.diag(A)
end


"""
    diagonal(A)

Get the diagonal of a 1d array of floats. Behaves like an identity function 
(returns itself). 

# Arguments

- A = 1d array of floats

# Value 

1d array of floats 

# Some notes

Originally intended for use when A is a 1 by 1 array, so may have unintended 
consequences for a 1d array of length > 1. 

"""
function diagonal(A::AbstractArray{Float64,1})
    
    return A
end


"""
    diagonal(A)

Get the diagonal of a single scalar (float) value. Behaves like an identity 
function (returns itself). 

# Arguments

- A = floating scalar

# Value 

Floating scalar

"""
function diagonal(A::Float64)
    
    return A
end


"""
    kron_diag(A, B)

Compute the diagonal of the Kronecker product of arrays or scalars

# Arguments

- A = square 2d array of floats, a 1d array of floats, or a scalar
- B = square 2d array of floats, a 1d array of floats, or a scalar

# Value 

2d array of floats

"""
function kron_diag(A, B)
    
    # Get the diagonals of A and B
    diagA = diagonal(A)
    diagB = diagonal(B)
    
    # Pre-allocate array to store resulting diagonal
    result = Array{Float64}(undef, length(diagB), length(diagA))
    # Iterate through all elements of diagonals and perform pairwise 
    # multiplication
    for i in 1:length(diagA) 
        result[:,i] .= diagA[i]*diagB
    end
    
    return result
end
