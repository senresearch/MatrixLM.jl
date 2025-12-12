


"""
    get_estimates(mlm_est::Mlm; level::Float64 = 0.95)

Return coefficient estimates and summaries for a fitted matrix
linear model. The function computes coefficient estimates, standard errors,
two-sided margin of error for the requested confidence level, t-statistics,
and the variance of the estimates.

# Arguments
- `mlm_est::Mlm`: fitted matrix linear model.
- `level::Float64`: confidence level for margins of error (default `0.95`).

# Returns
Tuple `(est_coef, me, tStatsOut, var_est)` where `est_coef` are the
coefficients, `me` are margins of error, `tStatsOut` are t-statistics, and
`var_est` are the variances of the coefficient estimates.
"""
function get_estimates(mlm_est::Mlm; level::Float64 = 0.95)

    est_coef = MatrixLM.coef(mlm_est)
    var_est = mlm_est.varB
    se = sqrt.(var_est)

    # Two-sided margin of error at the requested level
    me = se .* quantile(Normal(0, 1), (1.0 - level) / 2.0) .* -1.0
    
    tStatsOut = t_stat(mlm_est, true)
    
    return est_coef, me, tStatsOut, var_est
end
