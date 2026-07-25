using MatrixLM
using Documenter

# copy readme into index.md
open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    write(io, read(joinpath(@__DIR__, "..", "README.md"), String))
end

makedocs(; modules=[MatrixLM], sitename="MatrixLM.jl", pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Example: MLM for ordinal predictors" => "example_ordinal_data.md",
        "Example:Variance Shrinkage with MatrixLM.jl" => "varShrinkage_example.md",
        "Types and Functions" => "functions.md",
    ]
)

deploydocs(;
    repo= "github.com/senresearch/MatrixLM.jl.git",
    devbranch= "main",
    devurl = "dev"
)
