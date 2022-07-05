"""
    Response(Y::AbstractArray{Float64,2})

Type for storing response matrix

"""
mutable struct Response

    Y::AbstractArray{Float64,2} # Response matrix Y
end


"""
    Predictors(X::AbstractArray{Float64,2} ,  Z::AbstractArray{Float64,2} , hasXIntercept::Bool, hasZIntercept::Bool)

Type for storing predictor (covariate) matrices. Also stores boolean 
variables hasXIntercept and hasZIntercept (if they are not supplied, they 
default to false). 

"""
mutable struct Predictors
    
    # Assumes all categorical variables are coded in appropriate contrasts
    # Matrix of row covariates
    X::AbstractArray{Float64,2} 
    # Matrix of column covariates
    Z::AbstractArray{Float64,2} 
    
    # Boolean flag indicating whether X has an intercept
    hasXIntercept::Bool 
    # Boolean flag indicating whether Z has an intercept
    hasZIntercept::Bool 
    
    # Usual constructor
    Predictors(X, Z, hasXIntercept, hasZIntercept) = 
        new(X, Z, hasXIntercept, hasZIntercept)
    
    # Modified constructor that sets hasXIntercept and hasZIntercept to false 
    # by default
    Predictors(X, Z) = new(X, Z, false, false)
end


"""
    RawData(response::Response, predictors::Predictors)

Type for storing response and predictor matrices

Also stores dimensions of matrices as n, m, p, and q. 
- n = number of rows of X = number of rows of Y
- m = number of rows of Z = number of columns of Y
- p = number of columns of X
- q = number of columns of Z

The constructor will compute n, m, p, and q based on the response and 
predictor matrices and assert that they are consistent. 

"""
mutable struct RawData
    
    # Response
    response::Response 
    # Predictors
    predictors::Predictors

    # Data dimensions. 
    # These will be generated and should not be supplied as input to the 
    # constructor.
    n::Int64 # Rows of X = rows of Y
    m::Int64 # Rows of Z = columns of Y
    p::Int64 # Columns of X
    q::Int64 # Columns of Z

    RawData(response, predictors) = 
        # Assert that the dimensions in response and predictors match
        (size(predictors.X,1) != size(response.Y,1)) || 
            (size(predictors.Z,1) != size(response.Y,2)) ? 
        error("Dimension mismatch.") : 
        # Construct RawData object
        new(response, predictors, size(predictors.X,1), size(predictors.Z,1), 
            size(predictors.X,2), size(predictors.Z,2))
end


"""
    get_X(data::RawData)

Extract X matrix from RawData object 

# Arguments

- data = RawData object

# Value

2d array

"""
function get_X(data::RawData)
    
    return data.predictors.X
end


"""
    get_Z(data::RawData)

Extract Z matrix from RawData object

# Arguments

- data = RawData object

# Value

2d array

"""
function get_Z(data::RawData)
    
    return data.predictors.Z
end


"""
    get_Y(data::RawData)

Extract Y matrix from RawData object

# Arguments

- data = RawData object

# Value

2d array

"""
function get_Y(data::RawData)
    
    return data.response.Y
end
