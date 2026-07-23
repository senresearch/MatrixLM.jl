```@meta
ShareDefaultModule = true
```

# Variance Shrinkage

## Overview

In this section, we demonstrate how to use variance shrinkage when fitting a matrix linear model with `MatrixLM.jl`.

Within the matrix linear model framework,

$$
Y = XBZ^T + E,
$$

where

- ``Y_{n \times m}`` is the response matrix,
- ``X_{n \times p}`` is the row-predictor matrix,
- ``Z_{m \times q}`` is the response-attribute matrix,
- ``B_{p \times q}`` is the coefficient matrix, and
- ``E_{n \times m}`` is the error matrix.

The rows of ``E`` are assumed to have covariance matrix ``\Sigma`` across the ``m`` responses. Estimating ``\Sigma`` directly from the residuals can be unstable when the number of responses is large relative to the sample size.
Variance shrinkage stabilizes this estimate by shrinking noisy sample quantities toward structured targets using the analytic shrinkage procedures proposed by Schäfer and Strimmer (2005)[^1] and Opgen-Rhein and Strimmer (2007)[^2].

`MatrixLM.jl` provides a simple Boolean interface for this choice:

```julia
mlm(data, false)  # no variance shrinkage
mlm(data, true)   # use the a shrinkage estimator
```

When `varShrinkage=true`, `MatrixLM.jl` applies a shrinkage estimator, which separately shrinks transformed correlations and transformed variances toward their respective common targets.

## Data Generation

We begin by simulating data from a matrix linear model with correlated, heteroskedastic errors. This setting is useful for illustrating variance shrinkage because the response variables have both unequal variances and nonzero
correlations.

```@example varshrinkage
using MatrixLM, LinearAlgebra, Random, Statistics
Random.seed!(1)

# Matrix dimensions
n = 100
m = 250
p = 5
q = 4

# Row and column predictors
X = randn(n, p)
Z = randn(m, q)

# True coefficient matrix
B = [
     1.5   0.0   0.5  -1.0;
     0.0   1.0   0.0   0.5;
    -0.5   0.0   1.5   0.0;
     0.0  -1.0   0.5   1.0;
     1.0   0.5   0.0   0.0
]

nothing #hide
```

To illustrate variance shrinkage, we generate an error matrix with correlated responses and unequal response variances. Correlation is introduced through a common random effect shared by all responses, while each response is assigned its own variance.

```@example varshrinkage
# Unequal response variances
variances = range(0.5, 2.0, length=m)
standard_deviations = sqrt.(variances)

# Common correlation between responses
rho = 0.6

# Generate correlated, heteroskedastic errors
common_noise = randn(n, 1)
independent_noise = randn(n, m)

E = (
    sqrt(rho) .* common_noise .+
    sqrt(1 - rho) .* independent_noise
) .* reshape(standard_deviations, 1, m)

# True covariance matrix
R = fill(rho, m, m)
R[diagind(R)] .= 1.0
Dhalf = Diagonal(standard_deviations)
Sigma_true = Dhalf * R * Dhalf

# Generate the response matrix
Y = X * B * Z' + E

nothing #hide
```
Now construct a `RawData` object containing `Y`, `X`, and `Z`.

```@example varshrinkage
dat = RawData(
    Response(Y),
    Predictors(X, Z)
)

nothing #hide
```

## Model Estimation Without Variance Shrinkage

Set the positional Boolean argument to `false` to estimate the residual covariance matrix without shrinkage.

```@example varshrinkage
est_no_shrinkage = mlm(
    dat,
    false;
    addXIntercept=false,
    addZIntercept=false
)

nothing #hide
```

The fitted object contains the coefficient estimates in `B`, the estimated error covariance matrix in `sigma`, and the coefficient variance estimates in `varB`.

```@example varshrinkage
size(est_no_shrinkage.B), size(est_no_shrinkage.sigma),
size(est_no_shrinkage.varB)
```

## Model Estimation With Variance Shrinkage

Set the positional Boolean argument to `true` to use the recommended variance shrinkage estimator.

```@example varshrinkage
est_shrinkage = mlm(
    dat,
    true;
    addXIntercept=false,
    addZIntercept=false
)

nothing #hide
```

The estimator separately shrinks

1. Fisher-transformed sample correlations toward their common mean, and
2. log-transformed sample variances toward their common mean.

The fitted object records the estimated shrinkage intensities.

```@example varshrinkage
est_shrinkage.lambda
```

The returned `lambda` object contains separate shrinkage intensities for the correlation and variance components.

## Comparing the Two Fits

Variance shrinkage changes the estimated error covariance matrix and, consequently, the estimated variances and standard errors of the coefficients. It does not change the least-squares coefficient estimates.

```@example varshrinkage
maximum(abs.(est_no_shrinkage.B - est_shrinkage.B))
```

The covariance estimates generally differ:

```@example varshrinkage
covariance_difference =
    norm(est_no_shrinkage.sigma - est_shrinkage.sigma)

covariance_difference
```

The coefficient variance estimates can also differ because they depend on the estimated error covariance matrix:

```@example varshrinkage
coefficient_variance_difference =
    norm(est_no_shrinkage.varB - est_shrinkage.varB)

coefficient_variance_difference
```

For this simulated example, we can compare both covariance estimators with the known data-generating covariance matrix.

```@example varshrinkage
error_no_shrinkage =
    norm(est_no_shrinkage.sigma - Sigma_true)

error_shrinkage =
    norm(est_shrinkage.sigma - Sigma_true)

(
    no_shrinkage = error_no_shrinkage,
    R_shrinkage = error_shrinkage
)
```

## Standard Errors and T-statistics

The coefficient standard errors are obtained from `varB`.

```@example varshrinkage
se_no_shrinkage = sqrt.(est_no_shrinkage.varB)
se_shrinkage = sqrt.(est_shrinkage.varB)
```

The corresponding t-statistics can be obtained with `t_stat`.

```@example varshrinkage
tstats_no_shrinkage = t_stat(est_no_shrinkage)
tstats_shrinkage = t_stat(est_shrinkage)

nothing #hide
```

Because variance shrinkage affects `varB`, it may also affect t-statistics, confidence intervals, and hypothesis-testing results.

## Summary

Variance shrinkage can be enabled in `MatrixLM.jl` by passing a Boolean as the second positional argument to `mlm`:

```julia
est = mlm(data, true)
```

The two principal calls are:

```julia
mlm(data, false)  # no variance shrinkage
mlm(data, true)   # variance shrinkage
```

The estimator stabilizes the residual covariance estimate by separately shrinking transformed correlations and transformed variances. The resulting covariance estimate is stored in `est.sigma`, the coefficient variance estimates are stored in `est.varB`, and the estimated shrinkage intensities are stored in `est.lambda`.

## References

[^1]: Schäfer, J., & Strimmer, K. (2005). A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics. Statistical Applications in Genetics and Molecular Biology, 4(1). https://doi.org/10.2202/1544-6115.1175

[^2]: Opgen-Rhein, R., & Strimmer, K. (2007). Accurate Ranking of Differentially Expressed Genes by a Distribution-Free Shrinkage Approach. Statistical Applications in Genetics and Molecular Biology, 6(1). https://doi.org/10.2202/1544-6115.1252
