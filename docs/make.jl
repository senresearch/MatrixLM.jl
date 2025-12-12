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
        "Types and Functions" => "functions.md",
    ]
)

deploydocs(;
    repo= "https://github.com/senresearch/MatrixLM.jl",
    devbranch= "main",
    devurl = "stable"
)
