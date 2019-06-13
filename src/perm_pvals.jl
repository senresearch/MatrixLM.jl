"""
    perm_pvals(fun, data, nPerms; permFun, funArgs...)

Obtains permutation p-values. 

# Arguments

- fun = function that returns a test statistic
- data = RawData object
- nPerms = number of permutations. Defaults to `1000`.

# Keyword arguments

- permFun = function used to permute `Y`. Defaults to `shuffle_rows` 
  (shuffles rows of `Y`). 
- funArgs = variable keyword arguments to be passed into `fun`

# Value

Tuple
- testStats: 2d array of floats; t-statistics
- pvals: 2d array of floats; permutation p-values

# Some notes

Permutations are computed in parallel when possible. 

"""
function perm_pvals(fun::Function, data::RawData, nPerms::Int64=1000; 
                    permFun::Function=shuffle_rows, funArgs...)
    
    # Calculate test statistics for data
    testStats = fun(data; funArgs...)
    # Take the absolute value of the test statistics
    absTestStats= abs.(testStats)

    # Initialze array to store permutation p-values
    pvals = convert(SharedArrays.SharedArray{Float64,2}, 
                    zeros(size(testStats))) 
    # Initialize array to store test statistics based on permuted data
    absPermTestStats = Array{Float64}(undef, size(testStats))
    # Initialize RawData object to store permuted Y
    dataPerm = RawData(Response(zeros(size(get_Y(data)))), data.predictors)

    # For every permutation
    @sync @distributed for i = 1:nPerms
        # Permute Y
        dataPerm.response.Y[:,:] = permFun(get_Y(data))
        # Calculate test statistics based on permuted data
        absPermTestStats[:,:] = abs.(fun(dataPerm; funArgs...))
        
        # Increment p-values based on permuted test statistics
        for k in 1:length(pvals)
            pvals[k] += (absTestStats[k]<=absPermTestStats[k])/nPerms
        end
    end
    
    # Return test statistics and permutation p-values
    return testStats, convert(Array{Float64,2}, pvals)
end
