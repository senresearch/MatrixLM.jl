# MatrixLM

[![CI](https://github.com/senresearch/MatrixLM.jl/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/senresearch/MatrixLM.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/senresearch/MatrixLM.jl/branch/dev/graph/badge.svg?token=uHM6utUQoi)](https://codecov.io/gh/senresearch/MatrixLM.jl)
[![MIT license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/chenhz1223/MatrixLM.jl/blob/main/LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://senresearch.github.io/MatrixLM.jl/stable/)
[![Pkg Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


## Description

This package can estimates matrix linear models. The core functions to obtain closed-form least squares estimates for matrix linear models. Variance shrinkage is adapted from [Ledoit & Wolf (2003)](https://www.sciencedirect.com/science/article/pii/S0927539803000070).


An extension of `MatrixLM` for applications in high-throughput genetic screens is the [`GeneticScreens`](https://github.com/senresearch/GeneticScreens.jl) package. See the associated paper, ["Matrix linear models for high-throughput chemical genetic screens"](http://dx.doi.org/10.1534/genetics.119.302299), and its [reproducible code](https://github.com/senresearch/mlm_gs_supplement) for more details. 

[`MatrixLMnet`](https://github.com/senresearch/MatrixLMnet.jl) is a related package that implements algorithms for  L1-penalized estimates for matrix linear models. See the associated paper, ["Sparse matrix linear models for structured high-throughput data"](https://arxiv.org/abs/1712.05767), and its [reproducible code](https://github.com/senresearch/mlm_l1_supplement) for more details. 

## Installation 

The `MatrixLM` package can be installed by running: 

```
using Pkg
Pkg.add("MatrixLM")
```

For the most recent version, use:
```
using Pkg
Pkg.add(url = "https://github.com/senresearch/MatrixLM.jl", rev="master")
```
Alternatively, you can also install `MatrixLM` from the julia REPL. Press `]` to enter pkg mode again, and enter the following:

```
add MatrixLM
```

## Contributing

We appreciate contributions from users including reporting bugs, fixing
issues, improving performance and adding new features.


## Questions

If you have questions about contributing or using `matrixLM` package, please communicate author form github.

Additional details can be found in the documentation for specific functions. 
