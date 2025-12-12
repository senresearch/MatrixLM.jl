
using Random
using Distributions
using Statistics

@testset "get_estimates" begin
    tol = 1e-8

    # Small reproducible data so GLM comparison is cheap
    Random.seed!(123)
    n, m, p, q = 500, 8, 3, 2
    X = rand(n, p)
    Z = rand(m, q)
    B = randn(p, q)
    E = 0.1 .* randn(n, m)
    Y = X * B * transpose(Z) + E

    # Build GLM design matrix by vec(Y) ~ kron(Z, X)
    glm_X = kron(Z, X)
    glm_y = vec(Y)
    glm_fit = lm(glm_X, glm_y)

    # MLM fit
    raw = RawData(Response(Y), Predictors(X, Z))
    mlm_est = mlm(raw, addXIntercept = false, addZIntercept = false)

    est_coef, me95, tStatsOut, var_est = get_estimates(mlm_est; level = 0.95)

    # Expected values from MLM objects
    expected_coef = MatrixLM.coef(mlm_est)
    expected_var = mlm_est.varB
    expected_se = sqrt.(expected_var)
    expected_me95 = expected_se .* quantile(Normal(0, 1), (1.0 - 0.95) / 2.0) .* -1.0
    expected_t = MatrixLM.t_stat(mlm_est, true)

    # 1 - Internal consistency with MLM fields
    @test est_coef ≈ expected_coef atol = tol
    @test var_est ≈ expected_var atol = tol
    @test me95 ≈ expected_me95 atol = tol
    @test tStatsOut ≈ expected_t atol = tol
    @test all(me95 .>= 0.0)

    # 2 - Compare against GLM fit (coefficients and SEs) up to ordering
    glm_coef = GLM.coef(glm_fit)
    glm_se = GLM.stderror(glm_fit)

    # Flatten MLM outputs to match GLM vector ordering (vec along columns)
    mlm_coef_vec = vec(est_coef)
    mlm_se_vec = vec(expected_se)

    @test length(glm_coef) == length(mlm_coef_vec)
    @test mlm_coef_vec ≈ glm_coef atol = tol
    @test mlm_se_vec ≈ glm_se atol = 1e-3  # allow a bit more tolerance for SEs

    # 3 - Confidence interval consistency with GLM 95% (symmetric z CI)
    zcrit = quantile(Normal(0, 1), 0.975)
    mlm_ci_lower = mlm_coef_vec .- zcrit .* mlm_se_vec
    mlm_ci_upper = mlm_coef_vec .+ zcrit .* mlm_se_vec
    glm_ci = confint(glm_fit)  # columns: lower, upper

    @test round.(mlm_ci_lower, digits = 3) ≈ round.(glm_ci[:, 1], digits = 3) atol = 1e-2
    @test round.(mlm_ci_upper, digits = 3) ≈ round.(glm_ci[:, 2], digits = 3) atol = 1e-2

    # 4 - Margin of error shrinks at lower confidence level
    _, me90, _, _ = get_estimates(mlm_est; level = 0.90)
    @test all(me90 .< me95)
end

