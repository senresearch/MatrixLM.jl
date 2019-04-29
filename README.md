# matrixLM

Core functions for closed-form least squares estimates for matrix linear models. `matrixLM` was developed in [Julia v1.1](https://julialang.org/downloads/). 

An extension of `matrixLM` for applications in high-throughput genetic screens is the [`GeneticScreen`](https://github.com/janewliang/GeneticScreen.jl) package. See the associated paper, ["Matrix linear models for high-throughput chemical genetic screens"](https://www.biorxiv.org/content/10.1101/468140v1), for more details. 

## Installation 

The `matrixLM` package can be installed by running: 

```
using Pkg; Pkg.add("https://github.com/janewliang/matrixLM.jl")
```

## Usage 

```
using matrixLM
```