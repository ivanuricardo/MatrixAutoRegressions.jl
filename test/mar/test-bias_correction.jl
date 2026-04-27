
@testset "Bootstrap MAR" begin
    sims = 100
    n1, n2 = (3, 4)
    p = 2
    n = n1 * n2
    obs = 50
    dgp = simulate_mar(obs; n1, n2, p)
    true_coef = dgp.C

    boot = Bootstrap(n_boot=100)
    boot_unres = Bootstrap(n_boot=100, restricted=false)
    sum_bs_res = [zeros(n, n) for _ in 1:p]
    sum_bs_unres = [zeros(n, n) for _ in 1:p]
    sum_raw = [zeros(n, n) for _ in 1:p]

    @time for i in 1:sims
        dgp = simulate_mar(obs; n1, n2, p, A=dgp.A, B=dgp.B, Sigma1=dgp.Sigma1, Sigma2=dgp.Sigma2)
        mar_model = MAR(dgp.Y; p)
        fit!(mar_model)
        var_model = VAR(vectorize(dgp.Y); p)
        fit!(var_model)

        bs_res = bias_correction(mar_model, boot)
        bs_unres = bias_correction(var_model, boot)

        for j in 1:p
            sum_bs_res[j] .+= bs_res.C[j]
            sum_bs_unres[j] .+= bs_unres.C[j]
            sum_raw[j] .+= mar_model.C[j]
        end
    end

    for j in 1:p
        bias_raw = norm(sum_raw[j] / sims - true_coef[j])
        bias_bs_res = norm(sum_bs_res[j] / sims - true_coef[j])
        bias_bs_unres = norm(sum_bs_unres[j] / sims - true_coef[j])
        @test bias_bs_res < bias_raw
    end
end

