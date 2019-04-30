# matrixLM

Core functions for closed-form least squares estimates for matrix linear models. 

An extension of `matrixLM` for applications in high-throughput genetic screens is the [`GeneticScreen`](https://github.com/janewliang/GeneticScreen.jl) package. See the associated paper, ["Matrix linear models for high-throughput chemical genetic screens"](https://www.biorxiv.org/content/10.1101/468140v1), for more details. 

## Installation 

The `matrixLM` package can be installed by running: 

```
using Pkg; Pkg.add("https://github.com/janewliang/matrixLM.jl")
```

`matrixLM` was developed in [Julia v1.1](https://julialang.org/downloads/). 

## Usage 

```
using matrixLM
```

First, construct a `RawData` object consisting of the response variable `Y` and row/column predictors `X` and `Z`. 

```
using Random

# Dimensions of matrices 
n = 100
m = 250
p = 10
q = 20

# Randomly generate some data
Random.seed!(1)
X = rand(n,p)
Z = rand(m,q)
B = rand(1:20,p,q)
E = randn(n,m)
Y = X*B*transpose(Z)+E

# Construct a RawData object
dat = RawData(Response(Y), Predictors(X, Z))
```

Least-squares estimates for matrix linear models can be obtained by running `mlm`. An object of type `Mlm` will be returned, with variables for the coefficient estimates (`B`), coefficient variance estimates (`varB`), and the estimated variance of the erros (`sigma`) By default, `mlm` estimates both row and column main effects (X and Z intercepts), but this behavior can be suppressed by setting `isXIntercept = false` and/or `isZntercept = false`. Column weights for `Y` and the target type for variance shrinkage can be optionally supplied to `weights` and `targetType`, respectively. 

```
est = mlm(dat)
```

The coefficient estimates can be accessed using `coef(est)`. Predicted values and residuals can be obtained by calling `predict` and `resid`. By default, both of these functions use the same data used to fit the model. However, a new `Predictors` object can be passed into `predict` as the `newPredictors` argument and a new `RawData` object can be passed into `resid` as the `newData` argument. For convenience, `fitted(est)` will return fitted values by calling `predict` with the default arguments. 

```
preds = predict(est)
resids = resid(est)
```

The t-statistics for an `Mlm` object (defined as `est.B ./ sqrt.(est.varB)`) can be obtained by running `t_stat`. By default, `t_stat` does not return the corresponding t-statistics for any main effects that were estimated by `mlm`, but they will be returned if `isMainEff = true` is specified. 

```
tStats = t_stat(est)
```

Permutation p-values for the t-statistics can be computed by `mlm_perms`. `mlm_perms` calls the more general function `perm_pvals` and will run the permutations in parallel when possible. The illustrative example below only runs 5 permutations, but a different number can be specified as the second argument. By default, the function used to permute `Y` is `shuffle_rows`, which shuffles the rows for `Y`. Alternative functions to permute `Y`, such as `shuffle_cols`, can be passed into the argument `permFun`. `mlm_perms` calls `mlm` and `t_stat` , so the user is free to specify arguments for `mlm` or `t_stat`; by default, `mlm_perms` will call both functions using their default behavior. 

```
nPerms = 5
tStats, pVals = mlm_perms(dat, 5)
```

Additional details can be found in the documentation for specific functions. 