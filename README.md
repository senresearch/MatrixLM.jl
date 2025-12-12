# MatrixLM

[![CI](https://github.com/senresearch/MatrixLM.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/senresearch/MatrixLM.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/senresearch/MatrixLM.jl/branch/main/graph/badge.svg?token=uHM6utUQoi)](https://codecov.io/gh/senresearch/MatrixLM.jl)
[![MIT license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/senresearch/MatrixLM.jl/blob/main/LICENSE.md)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://senresearch.github.io/MatrixLM.jl/stable)
[![Pkg Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

## Description

This package provides functions for estimating matrix linear models
which are bilinear models of the form $$Y = X B Z^\prime + E,$$ where
$Y$ is the response matrix, $X$ and $Z$ are design matrices
contributing information on the rows and columns of the response
matrix, $B$ is a matrix of coefficients to be estimated and $E$ is
random errors.  The core functions obtain closed-form least squares
estimates for matrix linear models, and a sandwich estimator for the
variance.  Variance shrinkage is adapted from Ledoit & Wolf
(2003)[^1].

The package is intended for high-dimensional applications such as
high-throughput biological data.  The core functions are very fast as
they use matrix operations.  The package provides flexibility in
modeling via the use of model formulas for both row covariates ($X$)
and column covariates ($Z$).

The first application of `MatrixLM` was for high-throughput genetic
screens; some additional specialized functions are available in the
package
[`GeneticScreens`](https://github.com/senresearch/GeneticScreens.jl)
package. See the associated paper, ["Matrix linear models for
high-throughput chemical genetic
screens"](http://dx.doi.org/10.1534/genetics.119.302299), and its
[reproducible code](https://github.com/senresearch/mlm_gs_supplement)
for more details.

A second application is metabolomics.  See the associated paper,
["Matrix Linear Models for connecting metabolite composition to
individual characteristics"](https://www.mdpi.com/2218-1989/15/2/140)
and its associated [reproducible
code](https://github.com/senresearch/mlm-metabolomics-supplement) for
more details.

[`MatrixLMnet`](https://github.com/senresearch/MatrixLMnet.jl) is a
related package that provides sparse estimates of $B$ using L$_1$ and
L$_2$-penalized estimates. See the associated paper, ["Sparse matrix
linear models for structured high-throughput
data"](https://arxiv.org/abs/1712.05767), and its [reproducible
code](https://github.com/senresearch/mlm_l1_supplement) for more
details.

## Installation 

The `MatrixLM` package can be installed by running: 

```
using Pkg
Pkg.add("MatrixLM")
```

or from the julia REPL, press `]` to enter pkg mode, and execute the following command:

```
add MatrixLM
```

For the most recent (development) version, use:
```
using Pkg
Pkg.add(url = "https://github.com/senresearch/MatrixLM.jl", rev="main")
```

## Contributing

We appreciate contributions from users including reporting bugs, fixing issues, improving performance and adding new features.

## Questions

If you have questions about contributing or using `MatrixLM` package, please communicate with the authors via GitHub.

## Citing `MatrixLM`

If you use `MatrixLM` in a scientific publication, please consider citing the following paper:

Jane W Liang, Robert J Nichols, Śaunak Sen, Matrix Linear Models for High-Throughput Chemical Genetic Screens, Genetics, Volume 212, Issue 4, 1 August 2019, Pages 1063–1073, https://doi.org/10.1534/genetics.119.302299

```
@article{10.1534/genetics.119.302299,
    author = {Liang, Jane W and Nichols, Robert J and Sen, Śaunak},
    title = "{Matrix Linear Models for High-Throughput Chemical Genetic Screens}",
    journal = {Genetics},
    volume = {212},
    number = {4},
    pages = {1063-1073},
    year = {2019},
    month = {06},
    issn = {1943-2631},
    doi = {10.1534/genetics.119.302299},
    url = {https://doi.org/10.1534/genetics.119.302299},
    eprint = {https://academic.oup.com/genetics/article-pdf/212/4/1063/42105135/genetics1063.pdf},
}
```

## References

[^1]: Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of empirical finance, 10(5), 603-621. 
