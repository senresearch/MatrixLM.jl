"""
    mlm_perms(data::RawData, nPerms::Int64=1000; 
              permFun::Function=shuffle_rows, 
              addXIntercept::Bool=true, addZIntercept::Bool=true, 
              weights=nothing, targetType=nothing, isMainEff::Bool=false)

Obtains permutation p-values for MLM t-statistics. 

# Arguments

- `data::RawData`: RawData object
- `nPerms::Int64=1000`: Number of permutations. Defaults to `1000`.

# Keyword arguments

- permFun::Function: function used to permute `Y`. Defaults to `shuffle_rows` 
  (shuffles rows of `Y`). 
- addXIntercept::Bool=true: Boolean flag indicating whether or not to include an `X` 
  intercept (row main effects). Defaults to `true`. 
- addZIntercept::Bool=true: Boolean flag indicating whether or not to include a `Z` 
  intercept (column main effects). Defaults to `true`. 
- weights: 1d array of floats to use as column weights for `Y`, or `nothing`. 
  If the former, must be the same length as the number of columns of `Y`. 
  Defaults to `nothing`. 
- targetType: string indicating the target type toward which to shrink the 
  error variance, or `nothing`. If the former, acceptable inputs are "A", "B", 
  "C", and "D". Defaults to `nothing`.
    - "A": Target is identity matrix
    - "B": Target is diagonal matrix with constant diagonal
    - "C": Target is has same diagonal element, and same off-diagonal element
    - "D": Target is diagonal matrix with unequal entries 
- isMainEff::Bool: boolean flag indicating whether or not to include p-values for 
  the main effects

# Value

Tuple
- `tStats`: 2d array of floats; t-statistics
- `pvals`: 2d array of floats; permutation p-values

# Some notes

Permutations are computed in parallel when possible. 

"""
function mlm_perms(data::RawData, nPerms::Int64=1000; 
                   permFun::Function=shuffle_rows, 
                   addXIntercept::Bool=true, addZIntercept::Bool=true, 
                   weights=nothing, targetType=nothing, isMainEff::Bool=false) 
    
    # Wrapper function that performs MLM and gets t-statistics
    function mlm_t_stat(data::RawData)
        return t_stat(mlm(data; addXIntercept=addXIntercept, 
                      addZIntercept=addZIntercept, weights=weights, 
                      targetType=targetType), isMainEff)
    end
    
    # Get MLM t-statistcs and permutation p-values
    return perm_pvals(mlm_t_stat, data, nPerms; permFun=permFun)
end
