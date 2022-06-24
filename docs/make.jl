using Documenter
using MatrixLM

const src = "https://github.com/senresearch/MatrixLM.jl"

makedocs(
         sitename = "MatrixLM",
         authors = "Jane W. Liang, Saunak Sen",
         format = Documenter.HTML(),
         modules  = [MatrixLM],
         pages=[
                "Home" => "index.md",
                "Getting Started" => "starter.md",
                "User Guide" => "manual.md"
               ])
deploydocs(;
    repo="https://github.com/senresearch/MatrixLM.jl",
    devbranch= "main",
    devurl = "stable"
)