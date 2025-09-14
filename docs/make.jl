using MatrixAutoRegressions
using Documenter

DocMeta.setdocmeta!(MatrixAutoRegressions, :DocTestSetup, :(using MatrixAutoRegressions); recursive=true)

makedocs(;
    modules=[MatrixAutoRegressions],
    authors="Ivan Ricardo <iu.ricardo@maastrichtuniversity.nl> and contributors",
    sitename="MatrixAutoRegressions.jl",
    format=Documenter.HTML(;
        canonical="https://Ivan Ricardo.github.io/MatrixAutoRegressions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Ivan Ricardo/MatrixAutoRegressions.jl",
    devbranch="main",
)
