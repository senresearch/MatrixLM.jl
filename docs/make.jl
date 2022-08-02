using Documenter
using MatrixLM

const src = "https://github.com/senresearch/MatrixLM.jl"

makedocs(
         sitename = "MatrixLM",
         format = Documenter.HTML(),
         modules  = [MatrixLM],
         pages=[
                "Home" => "index.md",
                "Getting Started" => "Demo1_Simulation.md",
                "More examples" => "moreExamples.md",
                "Types and Functions" => "functions.md"
               ])
deploydocs(;
    repo= "https://github.com/senresearch/MatrixLM.jl",
    devbranch= "testing",
    devurl = "stable"
)