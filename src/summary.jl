
"""
    confint(mlm_est::Mlm; alpha::Float64 = 0.05)

Compute two-sided confidence intervals for the
estimated coefficients in a fitted matrix linear model.

# Arguments
- `mlm_est::Mlm`: fitted matrix linear model.
- `alpha::Float64`: two-sided significance level used to build the interval
    (default `0.05`, i.e. 95% confidence).

# Returns
DataFrame with columns `coefficient`, `ci_lower`, `ci_upper`, and `margin_of_error` for each
coefficient.
"""
function confint(mlm_est::Mlm; alpha::Float64 = 0.05)


    # Flatten MLM outputs to save into a dataframe
    est_coef = MatrixLM.coef(mlm_est) |> vec
    var_est = mlm_est.varB |> vec
    se = sqrt.(var_est)

    # Two-sided z critical value for the given significance level
    zcrit = quantile(Normal(0, 1), alpha / 2.0) .* -1.0
    me = se .* zcrit

    # Compute confidence interval bounds
    mlm_ci_lower = est_coef .- me
    mlm_ci_upper = est_coef .+ me

    return DataFrame(
        coefficient = est_coef,
        margin_of_error = me,
        ci_lower = mlm_ci_lower, 
        ci_upper = mlm_ci_upper,
    )
end


"""
    summary(mlm_est::Mlm; alpha::Float64 = 0.05,
                        permutation_test::Bool = false,
                        nPerms::Int = 500)

Summarize a fitted matrix linear model with coefficient estimates, standard
errors, two-sided confidence intervals, t-statistics, and p-values.

# Arguments
- `mlm_est::Mlm`: fitted matrix linear model.
- `alpha::Float64`: two-sided significance level used for confidence
    intervals and margins of error (default `0.05`, i.e. 95% confidence).
- `permutation_test::Bool`: whether to compute permutation-based t statistics
    and p-values using `mlm_perms`.
- `nPerms::Int`: number of permutations when `permutation_test` is `true`.

# Returns
DataFrame with columns `coef`, `std_error`, `t_stat`, `p_value`, `ci_lower`,
and `ci_upper`.
"""
function summary(mlm_est::Mlm; 
    alpha::Float64 = 0.05, permutation_test::Bool = false, nPerms::Int = 500)

    # Flatten MLM outputs to save into a dataframe
    est_coef = MatrixLM.coef(mlm_est) |> vec
    var_est = mlm_est.varB  |> vec
    se = sqrt.(var_est)

    # Get confidence intervals
    confint = MatrixLM.confint(mlm_est; alpha = alpha)

    # Get t-statistics and p-values
    if permutation_test == true
        tStatsOut, pvalues = mlm_perms(mlm_est.data, nPerms)        
    else
        tStatsOut = t_stat(mlm_est, true)
        pvalues = ccdf.(TDist(mlm_est.data.n - 1), abs.(tStatsOut)).*2
    end

    return DataFrame(
        coef = est_coef,
        std_error = se,
        t_stat = tStatsOut |> vec,
        p_value = pvalues |> vec,
        ci_lower = confint.ci_lower,
        ci_upper = confint.ci_upper,
    )
end
