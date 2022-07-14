using Test
using MatrixLM

@testset "MatrixLM" begin 
    include("testMlm.jl")
    include("miscHelperTesting.jl")
    include("shrinkSigmaTesting.jl")
    include("mlmPermsTesting.jl")
    include("predictTesting.jl")
end


# Tests for variance shrinkage, WLS
# Contrasts
# Permutations
