"""
    Mlm_plots(tStats::Matrix{Float64}, n::Int64, colNames::Matrix{String}})

Plot the t-statistics of the coefficients.
#Inputs:
tStats::Matrix{Float64}: a matrix of t-statistics
n::Int64: the number of rows in the t-statistics MatrixLM
colNames::Matrix{string}: the column names of the columns in the t-statistics matrix.

"""
struct MLMplots
    data::Matrix{Real}
    nrow::Int
    xticks::Matrix{String}
end

@recipe function f(x::MLMplots)
    mB = x.data
    nrow = x.nrow
    mticks = x.xticks
    V = mB[:,nrow]
    
    # return error message if the input arugment is different from `AbstractMatrix`
    typeof(mB) <: AbstractMatrix || error("Pass a Matrix as the arg to lineplot")
    # get size of input matrix
    #rows, cols = size(mB)
    
    # turn off the background grid
    grid := false                      
    
    title --> string("T-statistics of Coefficients Estimates")
    xlabel --> "Variables"
    ylabel --> "T-statistics"
    xrotation --> 20
    legend --> false
    label --> "y1"
    
    xticks := (collect(1:length(mticks)), mticks)
    # add a series for an error band
    @series begin
        seriestype := :path
        #primary := false
        linecolor --> :blue
        markershape --> :circle
        
        # return series data
        V
    end
    
end
