using Pkg

# Build docs in the docs-specific environment so documentation-only
# dependencies (for example StableRNGs used in examples) are available.
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

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
        "Example: Variance Shrinkage with MatrixLM.jl" => "varShrinkage_example.md",
        "Example: Application to real data" => "application_to_real_data.md",
        "Types and Functions" => "functions.md",
    ]
)

deploydocs(;
    repo= "github.com/senresearch/MatrixLM.jl.git",
    devbranch= "main",
    devurl = "dev"
)
