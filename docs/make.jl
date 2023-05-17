push!(LOAD_PATH,"../src/")

using MatrixLM
using Documenter

makedocs(
        modules = [MatrixLM],
        sitename = "MatrixLM.jl",
        # format=Documenter.HTML(
        #     prettyurls = get(ENV, "CI", "false") == "true",
        #     canonical = "https://senresearch.github.io/MatrixLM.jl",
        # ),       
        pages=[
            "Home" => "index.md",
            "Getting Started" => "getting_started.md",
            "More examples" => "moreExamples.md",
            "Types and Functions" => "functions.md"
        ],
)
deploydocs(;
    repo= "https://github.com/senresearch/MatrixLM.jl",
    devbranch= "main",
    devurl = "stable"
)