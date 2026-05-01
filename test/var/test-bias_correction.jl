
@testset "Pope-Kilian bias" begin
    sims = 10000
    p = 2
    n = 4
    obs = 50
    dgp = simulate_var(obs; p, n)
    true_coef = dgp.C
    true_Sigma = dgp.Sigma

    sum_before = [zeros(n, n) for _ in 1:p]
    sum_after = [zeros(n, n) for _ in 1:p]

    for i in 1:sims
        dgp = simulate_var(obs; p, n, C=true_coef, Sigma=true_Sigma)
        var_model = VAR(dgp.Y; p)
        fit!(var_model)

        for j in 1:p
            sum_before[j] .+= var_model.C[j]
        end

        bias_correction!(var_model, Analytical())

        for j in 1:p
            sum_after[j] .+= var_model.C[j]
        end
    end

    for j in 1:p
        bias_before = norm(sum_before[j] / sims - true_coef[j])
        bias_after = norm(sum_after[j] / sims - true_coef[j])
        @test bias_before > bias_after
    end
end

@testset "Bootstrap vs Analytical" begin
    sims = 1000
    p = 2
    n = 4
    obs = 50
    dgp = simulate_var(obs; p, n)
    true_coef = dgp.C
    true_Sigma = dgp.Sigma

    sum_analytic = [zeros(n, n) for _ in 1:p]
    sum_bs = [zeros(n, n) for _ in 1:p]
    sum_raw = [zeros(n, n) for _ in 1:p]

    for i in 1:sims
        dgp = simulate_var(obs; p, n, C=true_coef, Sigma=true_Sigma)
        var_model = VAR(dgp.Y; p)
        fit!(var_model)

        analytic = bias_correction(var_model, Analytical())
        bs = bias_correction(var_model, Bootstrap(bias_runs=100))

        for j in 1:p
            sum_analytic[j] .+= analytic.C[j]
            sum_bs[j] .+= bs.C[j]
            sum_raw[j] .+= var_model.C[j]
        end
    end

    for j in 1:p
        bias_raw = norm(sum_raw[j] / sims - true_coef[j])
        bias_analytic = norm(sum_analytic[j] / sims - true_coef[j])
        bias_bs = norm(sum_bs[j] / sims - true_coef[j])
        @test bias_analytic < bias_raw
        @test bias_bs < bias_raw
    end
end

