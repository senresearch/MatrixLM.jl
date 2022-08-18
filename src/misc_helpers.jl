"""
    add_intercept(A::AbstractArray{Float64,2})

Insert an intercept column (column of ones) at the beginning of a 2d array. 

# Arguments 

- `A::AbstractArray{Float64,2}`: 2d array of floats

# Value 

Returns A with an intercept column

"""
function add_intercept(A::AbstractArray{Float64,2})
    
    # Check that the first column isn't a column of ones
    if all(A[:,1].==1)
        print("The first column of this array is already a column of ones. Are you sure you wanted to add an intercept?")
    end
    
    # Concatenate a column of ones to the beginning of A
    return hcat(ones(size(A,1)), A)
end


"""
    remove_intercept(A::AbstractArray{Float64,2})

Remove the intercept column, assumed to be the first column of a 2d array. 

# Arguments 

- `A::AbstractArray{Float64,2}`: 2d array of floats

# Value 

Returns A without the intercept column

"""
function remove_intercept(A::AbstractArray{Float64,2})
    
    # Check that the first column is a column of ones
    if any(A[:,1].!=1)
        print("The first column of this array does not look like an intercept 
               since it is not a column of ones. Are you sure you wanted to 
               remove it?")
    end
    
    # Remove the first column of A
    return A[:,2:end]
end


"""
    shuffle_rows(A::AbstractArray{Float64,2})

Shuffles rows of a 2d array 

# Arguments 

- `A::AbstractArray{Float64,2}`: 2d array of floats

# Value

Returns A with rows shuffled

"""
function shuffle_rows(A::AbstractArray{Float64,2})
    
    return A[Random.shuffle(1:size(A,1)),:]
end


"""
    shuffle_cols(A::AbstractArray{Float64,2})

Shuffles columns of a 2d array 

# Arguments 

- `A::AbstractArray{Float64,2}`: 2d array of floats

# Value

Returns A with columns shuffled

"""
function shuffle_cols(A::AbstractArray{Float64,2})
    
    return A[:,Random.shuffle(1:size(A,2))]
end


function is_full_rank(A::AbstractArray{Float64,2})

    return rank(A) == size(A)[2]
end

function check_Z_rank(A::AbstractArray{Float64,2})
    if !is_full_rank(A)
        @warn "The rank of Z matrix is not full, and this may generate errores."
    end
end