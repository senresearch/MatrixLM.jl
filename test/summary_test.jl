using Random
using Distributions
using Statistics

@testset "confint and summary" begin
    tol = 1e-8

    # Small reproducible dataset
    Random.seed!(123)
    n, m, p, q = 200, 6, 3, 2
    X = rand(n, p)
    Z = rand(m, q)
    B = randn(p, q)
    E = 0.05 .* randn(n, m)
    Y = X * B * transpose(Z) + E

    # Build GLM design matrix by vec(Y) ~ kron(Z, X)
    glm_X = kron(Z, X)
    glm_y = vec(Y)
    glm_fit = lm(glm_X, glm_y)

    # MLM fit
    raw = RawData(Response(Y), Predictors(X, Z))
    mlm_est = mlm(raw, addXIntercept = false, addZIntercept = false)

    # Flatten MLM outputs to match GLM vector ordering (vec along columns)
    est_coef = MatrixLM.coef(mlm_est) |> vec
    var_est = mlm_est.varB |> vec
    se = sqrt.(var_est)

    alpha = 0.05  # 95% confidence
    zcrit = quantile(Normal(0, 1), alpha / 2.0) .* -1.0
    me_expected = se .* zcrit
    ci_lower_expected = est_coef .- me_expected
    ci_upper_expected = est_coef .+ me_expected

    # 1 - Confint checks
    ci_df = MatrixLM.confint(mlm_est; alpha = alpha)
    @test ci_df.coefficient ≈ est_coef atol = tol
    @test ci_df.margin_of_error ≈ me_expected atol = tol
    @test ci_df.ci_lower ≈ ci_lower_expected atol = tol
    @test ci_df.ci_upper ≈ ci_upper_expected atol = tol
    @test all(ci_df.margin_of_error .>= 0.0)

    # 2 - Summary checks 
    summ_df = MatrixLM.summary(mlm_est; alpha = alpha, permutation_test = false)
    t_exp = MatrixLM.t_stat(mlm_est, true) |> vec
    p_exp = ccdf.(TDist(mlm_est.data.n - 1), abs.(t_exp)) .* 2

    @test summ_df.coef ≈ est_coef atol = tol
    @test summ_df.std_error ≈ se atol = tol
    @test summ_df.t_stat ≈ t_exp atol = tol
    @test summ_df.p_value ≈ p_exp atol = 1e-6
    @test summ_df.ci_lower ≈ ci_df.ci_lower atol = tol
    @test summ_df.ci_upper ≈ ci_df.ci_upper atol = tol

    # 3 - Summary with permutation-based stats: just check shape and finiteness
    summ_perm = MatrixLM.summary(mlm_est; alpha = alpha, permutation_test = true, nPerms = 25)
    @test length(summ_perm.t_stat) == length(est_coef)
    @test length(summ_perm.p_value) == length(est_coef)
    @test all(isfinite, summ_perm.t_stat)
    @test all(x -> isfinite(x) && 0.0 <= x <= 1.0, summ_perm.p_value)

    # 4 - Compare against GLM fit (coefficients and SEs) up to ordering
    glm_coef = GLM.coef(glm_fit)
    glm_se = GLM.stderror(glm_fit)


    @test length(glm_coef) == length(est_coef)
    @test est_coef ≈ glm_coef atol = tol
    @test se ≈ glm_se atol = 1e-3  # allow a bit more tolerance for SEs

    # 5 - Confidence interval consistency with GLM 95% (symmetric z CI)
    glm_ci = GLM.confint(glm_fit)  # columns: lower, upper

    @test round.(ci_df.ci_lower, digits = 3) ≈ round.(glm_ci[:, 1], digits = 3) atol = 1e-2
    @test round.(ci_df.ci_upper, digits = 3) ≈ round.(glm_ci[:, 2], digits = 3) atol = 1e-2

end
