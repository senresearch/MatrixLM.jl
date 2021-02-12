# MatrixLM

Core functions to obtain closed-form least squares estimates for matrix linear models. Variance shrinkage is adapted from Ledoit & Wolf (2003)<sup>[1](#myfootnote1)</sup>.

An extension of `MatrixLM` for applications in high-throughput genetic screens is the [`GeneticScreens`](https://github.com/senresearch/GeneticScreens.jl) package. See the associated paper, ["Matrix linear models for high-throughput chemical genetic screens"](http://dx.doi.org/10.1534/genetics.119.302299), and its [reproducible code](https://github.com/senresearch/mlm_gs_supplement) for more details. 

[`MatrixLMnet`](https://github.com/senresearch/MatrixLMnet.jl) is a related package that implements algorithms for  L<sub>1</sub>-penalized estimates for matrix linear models. See the associated paper, ["Sparse matrix linear models for structured high-throughput data"](https://arxiv.org/abs/1712.05767), and its [reproducible code](https://github.com/senresearch/mlm_l1_supplement) for more details. 

## Installation 

The `MatrixLM` package can be installed by running: 

```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/senresearch/MatrixLM.jl", rev="master"))
```

`MatrixLM` was developed in [Julia v1.5.3](https://julialang.org/downloads/). 

## Usage 

```
using MatrixLM
```

First, construct a `RawData` object consisting of the response variable `Y` and row/column predictors `X` and `Z`. All three matrices must be passed in as 2-dimensional arrays. Note that the `contr` function can be used to set up treatment and/or sum contrasts for categorical variables stored in a DataFrame. By default, `contr` generates treatment contrasts for all specified categorical variables (`"treat"`). Other options include `"sum"` for sum contrasts, `"noint"` for treatment contrasts with no intercept, and `"sumnoint"` for sum contrasts with no intercept. 

```
using DataFrames
using Random

# Dimensions of matrices 
n = 100
m = 250
# Number of column covariates
q = 20

# Randomly generate an X matrix of row covariates with 2 categorical variables
# and 4 continuous variables
Random.seed!(1)
X_df = hcat(DataFrame(catvar1=rand(1:5, n), catvar2=rand(["A", "B", "C"], n)), 
            DataFrame(rand(n,4)))
# Use the contr function to get contrasts for the two categorical variables 
# (treatment contrasts for catvar1 and sum contrasts for catvar2).
# contr returns a DataFrame, so X needs to be converted into a 2d array.
X = convert(Array{Float64,2}, contr(X_df, [:catvar1, :catvar2], 
                                    ["treat", "sum"]))
# Number of row covariates
p = size(X)[2]

# Randomly generate some data for column covariates Z and response variable Y
Z = rand(m,q)
B = rand(-5:5,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E

# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z))
```

Least-squares estimates for matrix linear models can be obtained by running `mlm`. An object of type `Mlm` will be returned, with variables for the coefficient estimates (`B`), the coefficient variance estimates (`varB`), and the estimated variance of the errors (`sigma`). By default, `mlm` estimates both row and column main effects (X and Z intercepts), but this behavior can be suppressed by setting `isXIntercept=false` and/or `isZntercept=false`. Column weights for `Y` and the target type for variance shrinkage<sup>[1](#myfootnote1)</sup> can be optionally supplied to `weights` and `targetType`, respectively. 

```
est = mlm(dat)
```

The coefficient estimates can be accessed using `coef(est)`. Predicted values and residuals can be obtained by calling `predict` and `resid`. By default, both of these functions use the same data used to fit the model. However, a new `Predictors` object can be passed into `predict` as the `newPredictors` argument and a new `RawData` object can be passed into `resid` as the `newData` argument. For convenience, `fitted(est)` will return the fitted values by calling `predict` with the default arguments. 

```
preds = predict(est)
resids = resid(est)
```

The t-statistics for an `Mlm` object (defined as `est.B ./ sqrt.(est.varB)`) can be obtained by running `t_stat`. By default, `t_stat` does not return the corresponding t-statistics for any main effects that were estimated by `mlm`, but they will be returned if `isMainEff=true`. 

```
tStats = t_stat(est)
```

Permutation p-values for the t-statistics can be computed by `mlm_perms`. `mlm_perms` calls the more general function `perm_pvals` and will run the permutations in parallel when possible. The illustrative example below only runs 5 permutations, but a different number can be specified as the second argument. By default, the function used to permute `Y` is `shuffle_rows`, which shuffles the rows for `Y`. Alternative functions for permuting `Y`, such as `shuffle_cols`, can be passed into the argument `permFun`. `mlm_perms` calls `mlm` and `t_stat` , so the user is free to specify keyword arguments for `mlm` or `t_stat`; by default, `mlm_perms` will call both functions using their default behavior. 

```
nPerms = 5
tStats, pVals = mlm_perms(dat, nPerms)
```

Additional details can be found in the documentation for specific functions. 

---

<a name="myfootnote1">1</a>. Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of empirical finance, 10(5), 603-621. 
