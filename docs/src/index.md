# MatrixLM

[![Build Status](https://travis-ci.com/senresearch/MatrixLM.jl.svg?branch=master)](https://travis-ci.com/senresearch/MatrixLM.jl)
[![MIT license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/chenhz1223/MatrixLM.jl/blob/main/LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://senresearch.github.io/MatrixLM.jl/stable/)
[![Pkg Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

## Description

Core functions to obtain closed-form least squares estimates for matrix linear models. Variance shrinkage is adapted from [Ledoit & Wolf (2003)](https://www.sciencedirect.com/science/article/pii/S0927539803000070).


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

## Contributing

We appreciate contributions from users including reporting bugs, fixing
issues, improving performance and adding new features.

Take a look at the [contributing files](https://github.com/BioJulia/Contributing)
detailed contributor and maintainer guidelines, and code of conduct.


## Questions?

If you have a question about contributing or using BioJulia software, come
on over and chat to us on [the Julia Slack workspace](https://julialang.org/slack/), or you can try the
[Bio category of the Julia discourse site](https://discourse.julialang.org/c/domain/bio).

---

