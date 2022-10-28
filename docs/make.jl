using Documenter
using MatrixLM

const src = "https://github.com/senresearch/MatrixLM.jl"

makedocs(
        modules = [MatrixLM],
        repo="https://github.com/senresearch/MatrixLM.jl/blob/{commit}{path}#{line}",
        sitename = "MatrixLM.jl",
        format=Documenter.HTML(
            prettyurls = get(ENV, "CI", "false") == "true",
            canonical = "https://senresearch.github.io/MatrixLM.jl",
        ),       
        pages=[
            "Home" => "index.md",
            "Getting Started" => "getting_started.md",
            "More examples" => "moreExamples.md",
            "Types and Functions" => "functions.md"
        ],
)
deploydocs(;
    repo= "https://github.com/senresearch/MatrixLM.jl",
    devbranch= "testing",
    push_preview = true,
    devurl = "stable"
)