using Test
using MatrixLM

@testset "MatrixLM" begin 
    include("mlmTesting.jl")
    include("miscHelperTesting.jl")
    include("shrinkSigmaTesting.jl")
    include("mlmPermsTesting.jl")
    include("predictTesting.jl")
    include("kronDiagTesting.jl")
    include("designMatrixTesting.jl")
    include("contrTesting.jl")
    include("plotsTesting.jl")
end


# Tests for variance shrinkage, WLS
# Contrasts
# Permutations
