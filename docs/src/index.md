# MatrixLM

[![Build Status](https://travis-ci.com/senresearch/MatrixLM.jl.svg?branch=master)](https://travis-ci.com/senresearch/MatrixLM.jl)

Core functions to obtain closed-form least squares estimates for matrix linear models. Variance shrinkage is adapted from Ledoit & Wolf (2003) <sup>[1](#myfootnote1)</sup>.

An extension of `MatrixLM` for applications in high-throughput genetic screens is the [`GeneticScreens`](https://github.com/senresearch/GeneticScreens.jl) package. See the associated paper, ["Matrix linear models for high-throughput chemical genetic screens"](http://dx.doi.org/10.1534/genetics.119.302299), and its [reproducible code](https://github.com/senresearch/mlm_gs_supplement) for more details. 

[`MatrixLMnet`](https://github.com/senresearch/MatrixLMnet.jl) is a related package that implements algorithms for  L<sub>1</sub>-penalized estimates for matrix linear models. See the associated paper, ["Sparse matrix linear models for structured high-throughput data"](https://arxiv.org/abs/1712.05767), and its [reproducible code](https://github.com/senresearch/mlm_l1_supplement) for more details. 

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


---

<a name="myfootnote1">1</a>. Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of empirical finance, 10(5), 603-621. 